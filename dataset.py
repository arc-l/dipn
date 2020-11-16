from torch.utils.data.sampler import Sampler
import os
import numpy as np
import torch
import torch.utils.data
import cv2
import imutils
from torchvision.transforms import functional as TF
from PIL import Image
import torchvision
import random
from constants import is_real, workspace_limits, heightmap_resolution, PUSH_Q, GRASP_Q, color_mean, color_std, depth_mean, depth_std, used_binary_mean, used_binary_std, DEPTH_MIN, total_obj, colors_upper, colors_lower, binary_mean, binary_std, distance, resolution, resolution_pad, padding_width, resolution_crop
from math import atan2, cos, sin, sqrt, pi, degrees
import glob
import skimage.io
import json


class SegmentationDataset(torch.utils.data.Dataset):
    """
    Create segmentation dataset for training Mask R-CNN.
    One uses pre-defined color range to separate objects (assume the color in one image is unique).
    One directly reads masks.
    """

    def __init__(self, root, transforms, is_real=False):
        self.root = root
        self.transforms = transforms
        self.is_real = is_real
        # load all image files, sorting them to ensure that they are aligned
        self.color_imgs = list(sorted(os.listdir(os.path.join(root, "color-heightmaps"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth-heightmaps"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images
        color_path = os.path.join(self.root, "color-heightmaps", self.color_imgs[idx])
        # depth_path = os.path.join(self.root, "depth-heightmaps", self.depth_imgs[idx])

        # color image input
        color_img = cv2.imread(color_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # get masks
        hsv = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
        masks = []
        if self.is_real:
            gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.uint8)
            blurred = cv2.medianBlur(gray, 5)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                if cv2.contourArea(c) > 100:
                    mask = np.zeros(color_img.shape[:2], np.uint8)
                    cv2.drawContours(mask, [c], -1, (1), -1)
                    masks.append(mask)
                    # cv2.imshow('mask' + self.color_imgs[idx], mask)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
        else:
            for ci in range(1, np.max(mask_img)):
                mask = mask_img == ci
                if np.sum((mask == True)) > 100:
                    masks.append(mask)

        num_objs = len(masks)
        if num_objs > 0:
            masks = np.stack(masks, axis=0)

        # get bounding box coordinates for each mask
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            if xmin == xmax or ymin == ymax:
                num_objs = 0

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if num_objs > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.as_tensor([0], dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        num_objs = torch.tensor(num_objs)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["num_obj"] = num_objs

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img, target = self.transforms(color_img, target)

        return img, target

    def __len__(self):
        # return len(self.imgs)
        return len(self.color_imgs)


class ForegroundDataset(torch.utils.data.Dataset):
    '''
    Craete binary image, 1 means foreground, 0 means background.
    For grasp, we care about the center of object, while considering the clearance of gripper.
    For push, we know all pushs are from left to right.
    This labeling approach is the as the in the function get_neg of trainer.py
    '''

    def __init__(self, root, num_rotations, fine_tuning_num):
        self.root = root
        # load all image files, sorting them to ensure that they are aligned
        self.color_imgs = list(sorted(os.listdir(os.path.join(root, "color-heightmaps"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth-heightmaps"))))
        if fine_tuning_num is not None:
            self.color_imgs = self.color_imgs[:int(fine_tuning_num)]
            self.depth_imgs = self.depth_imgs[:int(fine_tuning_num)]
        self.num_rotations = num_rotations
        self.push_large_kernel = np.ones((61, 61))  # hyperparamter
        self.push_small_kernel = np.ones((15, 15))  # hyperparamter
        self.grasp_kernel = np.ones((9, 9))   # hyperparamter

    def __getitem__(self, idx):
        # load images
        color_path = os.path.join(self.root, "color-heightmaps", self.color_imgs[idx])
        depth_path = os.path.join(self.root, "depth-heightmaps", self.depth_imgs[idx])

        # color image input
        color_img = cv2.imread(color_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img_pil = Image.fromarray(color_img)

        # depth image input
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_img = depth_img.astype(np.float32) / 100000  # translate to meters
        depth_img_pil = Image.fromarray(depth_img)

        # binary push image target, we need boundary and some extra
        push_depth_img = np.copy(depth_img)
        push_depth_img[push_depth_img <= DEPTH_MIN] = 0
        push_depth_img[push_depth_img > DEPTH_MIN] = 1
        push_depth_large = cv2.filter2D(push_depth_img, -1, self.push_large_kernel)
        push_depth_large[push_depth_large < 1] = 0
        push_depth_large[push_depth_large > 1] = 1
        push_depth_small = cv2.filter2D(push_depth_img, -1, self.push_small_kernel)
        push_depth_small[push_depth_small < 1] = 0
        push_depth_small[push_depth_small > 1] = 1
        push_depth_final = push_depth_large - push_depth_small
        push_depth_final[push_depth_final < 0] = 0
        # prepare q values
        push_depth_final[push_depth_final == 1] = PUSH_Q
        push_depth_final[push_depth_final == 0] = 0
        target_push_img_pil = Image.fromarray(push_depth_final)

        # binary grasp image target, we need center part
        grasp_depth_img = np.copy(depth_img)
        grasp_depth_img[grasp_depth_img <= DEPTH_MIN] = -100
        grasp_depth_img[grasp_depth_img > DEPTH_MIN] = 1
        grasp_depth = cv2.filter2D(grasp_depth_img, -1, self.grasp_kernel)
        grasp_depth[grasp_depth < 1] = 0
        grasp_depth[grasp_depth > 1] = 1
        # prepare q values
        grasp_depth[grasp_depth == 1] = GRASP_Q
        grasp_depth[grasp_depth == 0] = 0
        target_grasp_img_pil = Image.fromarray(grasp_depth)

        color_img_pil, depth_img_pil, target_push_img_pil, target_grasp_img_pil = self.transforms(
            color_img_pil, depth_img_pil, target_push_img_pil, target_grasp_img_pil)

        return color_img_pil, depth_img_pil, target_push_img_pil, target_grasp_img_pil

    def __len__(self):
        return len(self.color_imgs)

    @torch.no_grad()
    def transforms(self, color_heightmap, depth_heightmap, target_push_heightmap, target_grasp_heightmap):

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap.size[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap.size[0]) / 2)
        color_heightmap_pad = TF.pad(color_heightmap, padding_width, fill=0, padding_mode='constant')
        depth_heightmap_pad = TF.pad(depth_heightmap, padding_width, fill=0, padding_mode='constant')
        depth_heightmap_pad_push = TF.pad(depth_heightmap, padding_width, fill=-1, padding_mode='constant')
        target_push_heightmap_pad = TF.pad(target_push_heightmap, padding_width, fill=0, padding_mode='constant')
        target_grasp_heightmap_pad = TF.pad(target_grasp_heightmap, padding_width, fill=0, padding_mode='constant')

        # Random rotate
        rotate_idx = random.randint(0, self.num_rotations - 1)
        rotate_theta = rotate_idx * (360 / self.num_rotations)
        color_heightmap_pad = TF.rotate(color_heightmap_pad, rotate_theta)
        depth_heightmap_pad = TF.rotate(depth_heightmap_pad, rotate_theta)
        depth_heightmap_pad_push = TF.rotate(depth_heightmap_pad_push, rotate_theta)
        target_push_heightmap_pad = TF.rotate(target_push_heightmap_pad, rotate_theta)
        target_grasp_heightmap_pad = TF.rotate(target_grasp_heightmap_pad, rotate_theta)

        color_heightmap_pad = np.array(color_heightmap_pad)
        depth_heightmap_pad = np.array(depth_heightmap_pad)
        depth_heightmap_pad_push = np.array(depth_heightmap_pad_push)
        target_push_heightmap_pad = np.array(target_push_heightmap_pad)
        target_grasp_heightmap_pad = np.array(target_grasp_heightmap_pad)
        # Post process for pushing, only pixel has something on the right (based
        # on heightmap) will be 1, otherwise it will be a empty push, also if the
        # pushed place is empty
        x_y_idx = np.argwhere(target_push_heightmap_pad > 0)
        # assume the gripper is 2 cm wide
        for idx in x_y_idx:
            x, y = tuple(idx)
            area = depth_heightmap_pad[max(0, x - 4):min(depth_heightmap_pad.shape[0], x + 5),
                                       min(depth_heightmap_pad.shape[1], y + 5):min(depth_heightmap_pad.shape[1], y + 30)]  # 5x40 in image size, so 5 cm of push
            if np.sum(area > DEPTH_MIN) == 0:
                target_push_heightmap_pad[x, y] = 0
            else:
                area = depth_heightmap_pad_push[max(0, x - 4):min(depth_heightmap_pad_push.shape[0], x + 5),
                                                min(depth_heightmap_pad_push.shape[1] - 1, y + 75)]
                if np.sum(area < 0) > 0:  # out of the workspace
                    target_push_heightmap_pad[x, y] = 0
        # Post process for grasping, only pixel has clearance on the left/right (based on heightmap) will be 1
        x_y_idx = np.argwhere(target_grasp_heightmap_pad > 0)
        for idx in x_y_idx:
            x, y = tuple(idx)
            if is_real:
                left_area = depth_heightmap_pad[max(0, x - 4):min(depth_heightmap_pad.shape[0], x + 5),
                                                max(0, y - 27):max(0, y - 22)]  # 2x3 pixels in each side
                right_area = depth_heightmap_pad[max(0, x - 4):min(depth_heightmap_pad.shape[0], x + 5),
                                                    min(depth_heightmap_pad.shape[1] - 1, y + 23):min(depth_heightmap_pad.shape[1], y + 28)]  # 2x3 pixels in each side
                if ((np.sum(left_area > DEPTH_MIN) > 0 and np.sum((left_area - depth_heightmap_pad[x, y]) > -0.05) > 0) or
                    (np.sum(right_area > DEPTH_MIN) > 0 and np.sum((right_area - depth_heightmap_pad[x, y]) > -0.05) > 0)):
                    target_grasp_heightmap_pad[x, y] = 0
            else:
                left_area = depth_heightmap_pad[max(0, x - 4):min(depth_heightmap_pad.shape[0], x + 5),
                                                max(0, y - 28):max(0, y - 18)]  # 2x3 pixels in each side
                right_area = depth_heightmap_pad[max(0, x - 4):min(depth_heightmap_pad.shape[0], x + 5),
                                                 min(depth_heightmap_pad.shape[1] - 1, y + 19):min(depth_heightmap_pad.shape[1], y + 29)]  # 2x3 pixels in each side
                if ((np.sum(left_area > DEPTH_MIN) > 0 and np.sum((left_area - depth_heightmap_pad[x, y]) > -0.04) > 0) or
                        (np.sum(right_area > DEPTH_MIN) > 0 and np.sum((right_area - depth_heightmap_pad[x, y]) > -0.04) > 0)):
                    target_grasp_heightmap_pad[x, y] = 0

        # To tensor
        color_heightmap_pad = TF.to_tensor(color_heightmap_pad)
        depth_heightmap_pad = TF.to_tensor(depth_heightmap_pad)
        target_push_heightmap_pad = TF.to_tensor(target_push_heightmap_pad)
        target_grasp_heightmap_pad = TF.to_tensor(target_grasp_heightmap_pad)

        # Normalize
        color_heightmap_pad = TF.normalize(color_heightmap_pad, color_mean, color_std, inplace=True)
        depth_heightmap_pad = TF.normalize(depth_heightmap_pad, depth_mean, depth_std, inplace=True)

        return color_heightmap_pad, depth_heightmap_pad, target_push_heightmap_pad, target_grasp_heightmap_pad


class PushPredictionMultiDatasetEvaluation(torch.utils.data.Dataset):
    '''
    Push Prediction Dataset for Evaluation
    Input: Image, Action (x, y), Pose (x, y)
    Output: Diff_x, Diff_y, Diff_angle
    '''

    def __init__(self, depth_imgs, actions, poses, binary_objs, is_padding=False):
        self.is_padding = is_padding
        self.total_obj = total_obj
        self.distance = distance
        self.workspace_limits = workspace_limits
        self.heightmap_resolution = heightmap_resolution

        self.prev_depth_imgs = []
        self.prev_poses = []
        self.actions = []
        self.binary_objs = []
        print('Total files', len(depth_imgs), len(actions), len(poses))

        for i in range(len(actions)):
            self.prev_depth_imgs.append(depth_imgs[i][48:272, 48:272])
            self.prev_poses.append(poses[i])
            self.actions.append(actions[i])
            self.binary_objs.append(binary_objs[i])

        print('Used files', len(self.prev_depth_imgs), len(self.prev_poses), len(self.actions), len(self.binary_objs))
        assert len(set([len(self.prev_depth_imgs), len(self.prev_poses),
                        len(self.actions), len(self.binary_objs)])) == 1

    def __getitem__(self, idx):
        # depth image input
        prev_depth_img = self.prev_depth_imgs[idx]
        # number of objects
        num_obj = len(self.prev_poses[idx])
        # poses
        prev_poses = torch.tensor(self.prev_poses[idx])
        # action
        action_start = torch.tensor(self.actions[idx])
        action_end = torch.tensor([action_start[0] + self.distance / 0.2, action_start[1]])

        # prev binary depth binary
        # obj
        prev_depth_binary_img_obj = np.copy(prev_depth_img)
        prev_depth_binary_img_obj[prev_depth_binary_img_obj <= DEPTH_MIN] = 0
        prev_depth_binary_img_obj[prev_depth_binary_img_obj > DEPTH_MIN] = 1
        temp = np.zeros((680, 680))
        temp[228:452, 228:452] = prev_depth_binary_img_obj
        prev_depth_binary_img_obj = temp[int(action_start[1] + 228) - 112:int(action_start[1] + 228) +
                                         112, int(action_start[0] + 228) - 40:int(action_start[0] + 228) + 184]
        # action
        prev_depth_binary_img_action = np.zeros_like(prev_depth_img)
        prev_depth_binary_img_action[int(action_start[1]) - 11:int(action_start[1]) +
                                     12, int(action_start[0]):int(action_end[0])] = 1
        temp = np.zeros((680, 680))
        temp[228:452, 228:452] = prev_depth_binary_img_action
        prev_depth_binary_img_action = temp[int(action_start[1] + 228) - 112:int(action_start[1] + 228) +
                                            112, int(action_start[0] + 228) - 40:int(action_start[0] + 228) + 184]

        binary_objs = self.binary_objs[idx]
        temp = np.zeros_like(binary_objs[:, :, 0:1])
        if self.is_padding:
            for _ in range(total_obj - num_obj):
                binary_objs = np.concatenate((binary_objs, temp), axis=-1)

        # centralize
        action_start_ori = torch.clone(action_start).detach()
        action_end_ori = torch.clone(action_end).detach()
        action_start[0] -= 40
        action_start[1] -= 112
        for pi in range(num_obj):
            prev_poses[pi] = prev_poses[pi] - action_start
        prev_poses = prev_poses.flatten()
        if self.is_padding:
            prev_poses = torch.cat((prev_poses, torch.tensor(
                [-100, -100] * (self.total_obj - num_obj), dtype=torch.float32)))
        action = torch.tensor([40.0, 112.0, 40.0 + self.distance / 0.2, 112.0])

        used_binary_img, binary_objs_total = self.transforms(
            prev_depth_binary_img_obj, prev_depth_binary_img_action, binary_objs)

        # sort input based on the distance to the action
        prev_poses_sort = prev_poses.numpy()
        if self.is_padding:
            prev_poses_sort = prev_poses_sort.reshape(self.total_obj, -1)
        else:
            prev_poses_sort = prev_poses_sort.reshape(num_obj, -1)
        action_start = np.array([40, 112])
        sort_idx = np.argsort(np.sum(np.square(prev_poses_sort - action_start), axis=1))
        prev_poses_sort = prev_poses_sort[sort_idx].flatten()
        prev_poses = torch.tensor(prev_poses_sort, dtype=torch.float32)
        binary_objs_total = binary_objs_total[sort_idx]

        return prev_poses, action, action_start_ori, action_end_ori, used_binary_img, binary_objs_total, num_obj, sort_idx

    def __len__(self):
        return len(self.actions)

    @torch.no_grad()
    def transforms(self, prev_depth_binary_img_obj, prev_depth_binary_img_action, binary_objs):
        prev_depth_binary_img_obj = TF.to_tensor(prev_depth_binary_img_obj)
        prev_depth_binary_img_action = TF.to_tensor(prev_depth_binary_img_action)
        used_binary_img = torch.cat((prev_depth_binary_img_obj, prev_depth_binary_img_action), dim=0)
        used_binary_img = TF.normalize(used_binary_img, used_binary_mean, used_binary_std, inplace=True)

        binary_objs_total = TF.to_tensor(binary_objs)
        current_binary_mean = binary_mean * binary_objs_total.size(0)
        current_binary_std = binary_std * binary_objs_total.size(0)
        binary_objs_total = TF.normalize(binary_objs_total, current_binary_mean, current_binary_std, inplace=True)

        return used_binary_img, binary_objs_total


class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for _, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_size]
            batch_lists.append(batches)

        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        # final flatten  - produce flat list of indexes
        lst = self.flatten_list(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class PushPredictionMultiDataset(torch.utils.data.Dataset):
    '''
    Push Prediction Dataset for training Push prediction network.
    The push distance is fixed, could be 5 or 10 cm.
    Track objects by color, so we assume each object has a unique color, however, this constraint does not needed in evalution.
    Input: Image, Action (x, y), Pose (x, y)
    Output: Diff_x, Diff_y, Diff_angle
    '''

    def __init__(self, root, distance, is_padding=False, cutoff=None):
        self.root = root
        self.total_obj = total_obj
        self.is_padding = is_padding
        # load all image files, sorting them to ensure that they are aligned
        color_imgs = list(sorted(os.listdir(os.path.join(root, "color-heightmaps"))))
        depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth-heightmaps"))))
        actions = list(sorted(os.listdir(os.path.join(root, "actions"))))
        poses = list(sorted(os.listdir(os.path.join(root, "poses"))))

        self.distance = distance
        self.workspace_limits = workspace_limits
        self.heightmap_resolution = heightmap_resolution

        self.prev_color_imgs = []
        self.prev_depth_imgs = []
        self.prev_poses = []
        self.actions = []
        self.next_color_imgs = []
        self.next_depth_imgs = []
        self.next_poses = []
        self.cluster_indices = {}
        print('Total files', len(color_imgs), len(depth_imgs), len(actions), len(poses))

        # only keep data that we will use, based on distance
        if self.distance == 10:
            for i in range(len(actions)):
                assert len(set([actions[i][:7], color_imgs[i][:7], depth_imgs[i][:7], poses[i][:7]])
                           ) == 1, (actions[i][:7], color_imgs[i][:7], depth_imgs[i][:7], poses[i][:7])
                if cutoff is not None:
                    if int(actions[i][:3]) > cutoff:
                        break

                if actions[i].endswith('00.action.txt'):
                    self.prev_color_imgs.append(color_imgs[i])
                    self.prev_depth_imgs.append(depth_imgs[i])
                    self.prev_poses.append(poses[i])
                    self.actions.append(actions[i])

                    # create cluster indices, so the the data with same amount of object will be put together
                    poses_path = os.path.join(self.root, "poses", poses[i])
                    with open(poses_path, 'r') as file:
                        filedata = file.read()
                        poses_str = filedata.split(' ')
                        num_obj = len(poses_str) // 5
                        if num_obj in self.cluster_indices:
                            self.cluster_indices[num_obj].append(len(self.prev_poses) - 1)
                        else:
                            self.cluster_indices[num_obj] = [len(self.prev_poses) - 1]
                elif actions[i].endswith('10.action.txt'):
                    self.next_color_imgs.append(color_imgs[i])
                    self.next_depth_imgs.append(depth_imgs[i])
                    self.next_poses.append(poses[i])

        if self.distance == 5:
            for i in range(len(actions)):
                assert len(set([actions[i][:7], color_imgs[i][:7], depth_imgs[i][:7], poses[i][:7]])
                            ) == 1, (actions[i][:7], color_imgs[i][:7], depth_imgs[i][:7], poses[i][:7])
                # only use a certain amount of data
                if cutoff is not None:
                    if i >= 11 * int(cutoff) * 100:
                        break

                # work in real world
                if actions[i].endswith('00.action.txt'):
                    self.prev_color_imgs.append(color_imgs[i])
                    self.prev_depth_imgs.append(depth_imgs[i])
                    self.prev_poses.append(poses[i])
                    self.actions.append(actions[i])

                    # create cluster indices, so the the data with same amount of object will be put together
                    poses_path = os.path.join(self.root, "poses", poses[i])
                    with open(poses_path, 'r') as file:
                        filedata = file.read()
                        poses_str = filedata.split(' ')
                        num_obj = len(poses_str) // 5
                        if num_obj in self.cluster_indices:
                            self.cluster_indices[num_obj].append(len(self.prev_poses) - 1)
                        else:
                            self.cluster_indices[num_obj] = [len(self.prev_poses) - 1]
                elif actions[i].endswith('5.action.txt'):
                    self.next_color_imgs.append(color_imgs[i])
                    self.next_depth_imgs.append(depth_imgs[i])
                    self.next_poses.append(poses[i])

                # work in sim, data accumulation, for a 10 cm push, we can have 0-5, 1-6, 2-7, ..., 5-10
                # if len(set([actions[i][:7], color_imgs[i][:7], depth_imgs[i][:7], poses[i][:7]])) != 1:
                #     print(actions[i][:7], color_imgs[i][:7], depth_imgs[i][:7], poses[i][:7])
                # if (actions[i].endswith('00.action.txt') or actions[i].endswith('01.action.txt') or actions[i].endswith('02.action.txt') or
                #         actions[i].endswith('03.action.txt') or actions[i].endswith('04.action.txt')):
                #     self.prev_color_imgs.append(color_imgs[i])
                #     self.prev_depth_imgs.append(depth_imgs[i])
                #     self.prev_poses.append(poses[i])
                #     self.actions.append(actions[i])

                #     # create cluster indices, so the the data with same amount of object will be put together
                #     poses_path = os.path.join(self.root, "poses", poses[i])
                #     with open(poses_path, 'r') as file:
                #         filedata = file.read()
                #         poses_str = filedata.split(' ')
                #         num_obj = len(poses_str) // 5
                #         if num_obj in self.cluster_indices:
                #             self.cluster_indices[num_obj].append(len(self.prev_poses) - 1)
                #         else:
                #             self.cluster_indices[num_obj] = [len(self.prev_poses) - 1]
                # elif (actions[i].endswith('06.action.txt') or actions[i].endswith('07.action.txt') or actions[i].endswith('08.action.txt') or
                #         actions[i].endswith('09.action.txt') or actions[i].endswith('10.action.txt')):
                #     self.next_color_imgs.append(color_imgs[i])
                #     self.next_depth_imgs.append(depth_imgs[i])
                #     self.next_poses.append(poses[i])
                # else:
                #     self.prev_color_imgs.append(color_imgs[i])
                #     self.prev_depth_imgs.append(depth_imgs[i])
                #     self.prev_poses.append(poses[i])
                #     self.actions.append(actions[i])
                #     self.next_color_imgs.append(color_imgs[i])
                #     self.next_depth_imgs.append(depth_imgs[i])
                #     self.next_poses.append(poses[i])

                #     # create cluster indices, so the the data with same amount of object will be put together
                #     poses_path = os.path.join(self.root, "poses", poses[i])
                #     with open(poses_path, 'r') as file:
                #         filedata = file.read()
                #         poses_str = filedata.split(' ')
                #         num_obj = len(poses_str) // 5
                #         if num_obj in self.cluster_indices:
                #             self.cluster_indices[num_obj].append(len(self.prev_poses) - 1)
                #         else:
                #             self.cluster_indices[num_obj] = [len(self.prev_poses) - 1]

        print('Used files', len(self.prev_color_imgs), len(self.next_color_imgs), len(self.prev_depth_imgs), len(self.next_depth_imgs),
              len(self.prev_poses), len(self.next_poses), len(self.actions))
        assert len(set([len(self.prev_color_imgs), len(self.next_color_imgs), len(self.prev_depth_imgs), len(self.next_depth_imgs),
                        len(self.prev_poses), len(self.next_poses), len(self.actions)])) == 1

    def __getitem__(self, idx):
        # load data path
        prev_color_path = os.path.join(self.root, "color-heightmaps", self.prev_color_imgs[idx])
        prev_depth_path = os.path.join(self.root, "depth-heightmaps", self.prev_depth_imgs[idx])
        prev_poses_path = os.path.join(self.root, "poses", self.prev_poses[idx])
        actions_path = os.path.join(self.root, "actions", self.actions[idx])
        next_color_path = os.path.join(self.root, "color-heightmaps", self.next_color_imgs[idx])
        next_depth_path = os.path.join(self.root, "depth-heightmaps", self.next_depth_imgs[idx])
        next_poses_path = os.path.join(self.root, "poses", self.next_poses[idx])

        # color image input
        prev_color_img = cv2.imread(prev_color_path)
        prev_color_img = cv2.cvtColor(prev_color_img, cv2.COLOR_BGR2RGB)
        next_color_img = cv2.imread(next_color_path)
        next_color_img = cv2.cvtColor(next_color_img, cv2.COLOR_BGR2RGB)

        # depth image input
        prev_depth_img = cv2.imread(prev_depth_path, cv2.IMREAD_UNCHANGED)
        prev_depth_img = prev_depth_img.astype(np.float32) / 100000  # translate to meters 100000
        next_depth_img = cv2.imread(next_depth_path, cv2.IMREAD_UNCHANGED)
        next_depth_img = next_depth_img.astype(np.float32) / 100000  # translate to meters 100000
        next_depth_img[next_depth_img < 0] = 0

        # poses
        with open(prev_poses_path, 'r') as file:
            filedata = file.read()
            poses = filedata.split(' ')
            num_obj = len(poses) // 5
            prev_poses = []
            for pi in range(num_obj):
                x = (float(poses[pi * 5]) - self.workspace_limits[0][0]) / self.heightmap_resolution
                y = (float(poses[pi * 5 + 1]) - self.workspace_limits[1][0]) / self.heightmap_resolution
                angle_y = degrees(float(poses[pi * 5 + 4]))
                prev_poses.extend([x, y, angle_y])
            prev_poses = torch.tensor(prev_poses)
        with open(next_poses_path, 'r') as file:
            filedata = file.read()
            poses = filedata.split(' ')
            assert len(poses) // 5 == num_obj
            next_poses = []
            for pi in range(num_obj):
                x = (float(poses[pi * 5]) - self.workspace_limits[0][0]) / self.heightmap_resolution
                y = (float(poses[pi * 5 + 1]) - self.workspace_limits[1][0]) / self.heightmap_resolution
                angle_y = degrees(float(poses[pi * 5 + 4]))
                next_poses.extend([x, y, angle_y])
            next_poses = torch.tensor(next_poses)

        # action
        with open(actions_path, 'r') as file:
            filedata = file.read()
            x, y = filedata.split(' ')
            x = (float(x) - self.workspace_limits[0][0]) / self.heightmap_resolution
            y = (float(y) - self.workspace_limits[1][0]) / self.heightmap_resolution
            action_start = torch.tensor([float(x), float(y)])
            action_end = torch.tensor([float(x + self.distance / 0.2), float(y)])

        # prev binary depth binary
        # obj
        prev_depth_binary_img_obj = np.copy(prev_depth_img)
        prev_depth_binary_img_obj[prev_depth_binary_img_obj <= DEPTH_MIN] = 0
        prev_depth_binary_img_obj[prev_depth_binary_img_obj > DEPTH_MIN] = 1
        temp = np.zeros((680, 680))
        temp[228:452, 228:452] = prev_depth_binary_img_obj
        prev_depth_binary_img_obj = temp[int(action_start[1] + 228) - 112:int(action_start[1] + 228) +
                                         112, int(action_start[0] + 228) - 40:int(action_start[0] + 228) + 184]

        # action
        prev_depth_binary_img_action = np.zeros_like(prev_depth_img)
        prev_depth_binary_img_action[int(action_start[1]) - 11:int(action_start[1]) +
                                     12, int(action_start[0]):int(action_end[0])] = 1
        temp = np.zeros((680, 680))
        temp[228:452, 228:452] = prev_depth_binary_img_action
        prev_depth_binary_img_action = temp[int(action_start[1] + 228) - 112:int(action_start[1] + 228) +
                                            112, int(action_start[0] + 228) - 40:int(action_start[0] + 228) + 184]

        # TODO: assume pose in order of blue, green, brown, orange, yellow
        imgcolor = np.copy(prev_color_img)
        imgcolor = imgcolor.astype(np.uint8)
        temp = np.zeros((480, 480, 3), dtype=np.uint8)
        temp[128:(480 - 128), 128:(480 - 128), :] = imgcolor
        imgcolor = cv2.cvtColor(temp, cv2.COLOR_RGB2HSV)
        binary_objs = []
        for ci in range(num_obj):
            crop = imgcolor[int(prev_poses[ci * 3 + 1]) + 128 - 30:int(prev_poses[ci * 3 + 1]) +
                            128 + 30, int(prev_poses[ci * 3]) + 128 - 30:int(prev_poses[ci * 3]) + 128 + 30, :]
            assert crop.shape[0] == 60 and crop.shape[1] == 60, (self.prev_color_imgs[idx], crop.shape)
            mask = cv2.inRange(crop, colors_lower[ci], colors_upper[ci])
            binary_objs.append(mask)

        # delta poses
        deltas = []
        for pi in range(num_obj):
            d_x = next_poses[pi * 3] - prev_poses[pi * 3]
            d_y = next_poses[pi * 3 + 1] - prev_poses[pi * 3 + 1]
            d_a = -(next_poses[pi * 3 + 2] - prev_poses[pi * 3 + 2])
            if d_a < -180:
                d_a = 360 + d_a
            elif d_a > 180:
                d_a = d_a - 360
            assert abs(d_a) < 120, (pi, d_a, self.prev_color_imgs[idx],
                                    self.next_color_imgs[idx], prev_poses, next_poses)
            deltas.extend([d_x, d_y, d_a])
        if self.is_padding:
            deltas.extend([0, 0, 0] * (self.total_obj - num_obj))
        deltas = torch.tensor(deltas, dtype=torch.float32)

        # centralize
        action_start_ori = torch.clone(action_start).detach()
        action_end_ori = torch.clone(action_end).detach()
        action_start[0] -= 40
        action_start[1] -= 112
        for pi in range(num_obj):
            prev_poses[pi * 3: pi * 3 + 2] = prev_poses[pi * 3: pi * 3 + 2] - action_start
            next_poses[pi * 3: pi * 3 + 2] = next_poses[pi * 3: pi * 3 + 2] - action_start
        prev_poses_no_angle = []
        for pi in range(num_obj):
            prev_poses_no_angle.extend([prev_poses[pi * 3], prev_poses[pi * 3 + 1]])
        if self.is_padding:
            prev_poses_no_angle.extend([-100, -100] * (self.total_obj - num_obj))
        next_poses_no_angle = []
        for pi in range(num_obj):
            next_poses_no_angle.extend([next_poses[pi * 3], next_poses[pi * 3 + 1]])
        if self.is_padding:
            next_poses_no_angle.extend([-100, -100] * (self.total_obj - num_obj))
        prev_poses = torch.tensor(prev_poses_no_angle, dtype=torch.float32)
        next_poses = torch.tensor(next_poses_no_angle, dtype=torch.float32)
        action = torch.tensor([40.0, 112.0, 40.0 + self.distance / 0.2, 112.0])

        prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, binary_objs_total = self.transforms(
            prev_color_img, prev_depth_img, next_color_img, next_depth_img, prev_depth_binary_img_obj, prev_depth_binary_img_action, binary_objs)

        # sort input based on the distance to the action
        prev_poses_sort = prev_poses.numpy()
        if self.is_padding:
            prev_poses_sort = prev_poses_sort.reshape(self.total_obj, -1)
        else:
            prev_poses_sort = prev_poses_sort.reshape(num_obj, -1)
        action_start = np.array([40, 112])
        sort_idx = np.argsort(np.sum(np.square(prev_poses_sort - action_start), axis=1))
        prev_poses_sort = prev_poses_sort[sort_idx].flatten()
        prev_poses = torch.tensor(prev_poses_sort, dtype=torch.float32)
        next_poses_sort = next_poses.numpy()
        if self.is_padding:
            next_poses_sort = next_poses_sort.reshape(self.total_obj, -1)
        else:
            next_poses_sort = next_poses_sort.reshape(num_obj, -1)
        next_poses_sort = next_poses_sort[sort_idx].flatten()
        next_poses = torch.tensor(next_poses_sort, dtype=torch.float32)
        deltas_sort = deltas.numpy()
        if self.is_padding:
            deltas_sort = deltas_sort.reshape(self.total_obj, -1)
        else:
            deltas_sort = deltas_sort.reshape(num_obj, -1)
        deltas_sort = deltas_sort[sort_idx].flatten()
        deltas = torch.tensor(deltas_sort, dtype=torch.float32)
        binary_objs_total = binary_objs_total[sort_idx]

        return prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, deltas, self.prev_color_imgs[
            idx], self.next_color_imgs[idx], action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx

    def __len__(self):
        return len(self.actions)

    @torch.no_grad()
    def transforms(self, prev_color_img, prev_depth_img, next_color_img, next_depth_img,
                   prev_depth_binary_img_obj, prev_depth_binary_img_action, binary_objs):
        # To tensor
        prev_color_img = TF.to_tensor(prev_color_img)
        prev_depth_img = TF.to_tensor(prev_depth_img)
        next_color_img = TF.to_tensor(next_color_img)
        next_depth_img = TF.to_tensor(next_depth_img)
        prev_depth_binary_img_obj = TF.to_tensor(prev_depth_binary_img_obj)
        prev_depth_binary_img_action = TF.to_tensor(prev_depth_binary_img_action)
        used_binary_img = torch.cat((prev_depth_binary_img_obj, prev_depth_binary_img_action), dim=0)
        used_binary_img = TF.normalize(used_binary_img, used_binary_mean, used_binary_std, inplace=True)
        binary_objs_total = TF.to_tensor(binary_objs[0])
        for ci in range(1, len(binary_objs)):
            temp = TF.to_tensor(binary_objs[ci])
            temp = TF.normalize(temp, binary_mean, binary_std, inplace=True)
            binary_objs_total = torch.cat((binary_objs_total, temp), dim=0)
        if self.is_padding:
            if len(binary_objs) < self.total_obj:
                temp = np.zeros_like(binary_objs[0])
                temp = TF.to_tensor(temp)
                temp = TF.normalize(temp, binary_mean, binary_std, inplace=True)
                for ci in range(len(binary_objs), self.total_obj):
                    binary_objs_total = torch.cat((binary_objs_total, temp), dim=0)

        # Normalize
        prev_color_img = TF.normalize(prev_color_img, color_mean, color_std, inplace=True)
        next_color_img = TF.normalize(next_color_img, color_mean, color_std, inplace=True)
        prev_depth_img = TF.normalize(prev_depth_img, depth_mean, depth_std, inplace=True)
        next_depth_img = TF.normalize(next_depth_img, depth_mean, depth_std, inplace=True)

        return prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, binary_objs_total



def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)


def getCenterOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    # angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    angle = atan2(eigenvectors[1, 1], eigenvectors[1, 0])  # orientation in radians
    return cntr[0], cntr[1], angle


def process_image(depth_image):
    # Convert depth image to binary
    img = np.copy(depth_image)
    img[img <= DEPTH_MIN] = 0
    img[img > DEPTH_MIN] = 255
    # img = cv2.blur(img,(5,5))
    # img[img > 100] = 255
    # img[img <= 100] = 0
    img = np.squeeze(img.astype(np.uint8))

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    objects = []
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 50 or 5000 < area:
            continue
        # Draw each contour only for visualisation purposes
        # cv2.drawContours(img, contours, i, (0, 0, 255), 2)
        # Find the orientation of each shape
        x, y, angle = getCenterOrientation(c, img)
        objects.extend([x, y, angle])
        return objects, img


def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                            batch_size=16,
                            num_workers=6,
                            shuffle=False)

    mean = 0.0
    for images, _, _, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)
    print(mean)

    var = 0.0
    # color image, depth image, target
    for images, _, _, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*320*320))
    print(std)

    # loader = torch.utils.data.DataLoader(dataset,
    #                                      batch_size=1,
    #                                      num_workers=7,
    #                                      shuffle=False)

    # mean = 0.0
    # count = 0
    # for prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx in loader:
    #     for n in range(int(num_obj[0])):
    #         image = binary_objs_total[:, n:n + 1, :, :]
    #         batch_samples = image.size(0)
    #         image = image.view(batch_samples, image.size(1), -1)
    #         mean += image.mean(2).sum(0)
    #         count += 1
    # mean = mean / count
    # print(mean)

    # var = 0.0
    # count = 0
    # # color image, depth image, target
    # for prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx in loader:
    #     for n in range(int(num_obj[0])):
    #         image = binary_objs_total[:, n:n + 1, :, :]
    #         batch_samples = image.size(0)
    #         image = image.view(batch_samples, image.size(1), -1)
    #         var += ((image - mean.unsqueeze(1))**2).sum([0, 2])
    #         count += 1
    # std = torch.sqrt(var / (count * 60 * 60))
    # print(std)

    # mean = 0.0
    # for prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx in loader:
    #     image = used_binary_img
    #     batch_samples = image.size(0)
    #     image = image.view(batch_samples, image.size(1), -1)
    #     mean += image.mean(2).sum(0)
    # mean = mean / len(loader.dataset)
    # print(mean)
    # var = 0.0
    # # color image, depth image, target
    # for prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx in loader:
    #     image = used_binary_img
    #     batch_samples = image.size(0)
    #     image = image.view(batch_samples, image.size(1), -1)
    #     var += ((image - mean.unsqueeze(1))**2).sum([0, 2])
    # std = torch.sqrt(var / (len(loader.dataset) * 224 * 224))
    # print(std)


if __name__ == "__main__":
    dataset_root = 'logs/final-pretrain/data'
    dataset = ForegroundDataset(dataset_root, 16)

    # dataset_root = 'logs_push/push/data'
    # dataset = PushPredictionMultiDatasetTest(dataset_root, 10)
    compute_mean_std(dataset)
