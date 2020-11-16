import cv2
import imutils
import math
from constants import is_real, workspace_limits, DEPTH_MIN, colors_lower, colors_upper, resolution_pad, resolution, resolution_crop, padding_width, heightmap_resolution, distance
import random
import numpy as np
from scipy import spatial
import torch
from dataset import PushPredictionMultiDatasetEvaluation
from push_net import PushPredictionNet
import os
import multiprocessing as mp
from trainer import Trainer
from PIL import Image
from train_maskrcnn import get_model_instance_segmentation
from torchvision.transforms import functional as TF
import copy

class Predictor():
    """
    Predict and generate images after push actions.
    Assume the color image and depth image are well matched.
    We use the masks to generate new images, so the quality of mask is important.
    The input to this forward function should be returned from the sample_actions.
    """

    def __init__(self, snapshot):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        push_model = PushPredictionNet()
        state = torch.load(snapshot)
        push_model.load_state_dict(state)
        self.push_model = push_model.to(self.device)
        self.push_model.eval()

    # only rotated_color_image, rotated_depth_image are padding to 320x320
    @torch.no_grad()
    def forward(self, rotated_color_image, rotated_depth_image, rotated_action, rotated_center,
                rotated_angle, rotated_binary_objs, rotated_mask_objs, plot=False):
        # get data
        dataset = PushPredictionMultiDatasetEvaluation(
            rotated_depth_image, rotated_action, rotated_center, rotated_binary_objs)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(rotated_depth_image), shuffle=False, num_workers=7)
        prev_poses, action, action_start_ori, action_end_ori, used_binary_img, binary_objs_total, num_obj, sort_idx = next(iter(data_loader))
        prev_poses = prev_poses.to(self.device, non_blocking=True)
        used_binary_img = used_binary_img.to(self.device, non_blocking=True, dtype=torch.float)
        binary_objs_total = binary_objs_total.to(self.device, non_blocking=True)
        action = action.to(self.device, non_blocking=True)
        # get output
        output = self.push_model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
        output = output.cpu().numpy()

        # generate new images
        prev_poses_input = prev_poses.cpu().numpy().astype(int)
        prev_poses = copy.deepcopy(prev_poses_input)
        action_start_ori = action_start_ori.numpy().astype(int)
        action_end_ori = action_end_ori.numpy().astype(int)
        action_start_ori_tile = np.tile(action_start_ori, num_obj[0])
        action_start = action[:, :2].cpu().numpy().astype(int)
        action_start_tile = np.tile(action_start, num_obj[0])
        sort_idx = sort_idx.cpu().numpy().astype(int)
        generated_color_images = []
        generated_depth_images = []
        validations = []

        for i in range(len(rotated_depth_image)):
            i_output = output[i]
            i_prev_poses = prev_poses[i]
            i_action_start_ori_tile = action_start_ori_tile[i]
            i_action_start_tile = action_start_tile[i]
            i_prev_poses += i_action_start_ori_tile
            i_prev_poses -= i_action_start_tile
            i_rotated_angle = rotated_angle[i]
            i_rotated_mask_objs = rotated_mask_objs[i]
            i_sort_idx = sort_idx[i]
            color_image = rotated_color_image[i]
            depth_image = rotated_depth_image[i]
            # transform points and fill them into a black image
            generated_color_image = np.zeros_like(color_image)
            generated_depth_image = np.zeros_like(depth_image)
            post_points_pad = []
            post_new_points_pad = []

            # for each object
            valid = True
            for pi in range(num_obj[i]):
                # if the object is out of the boundary, then, we can skip this action
                if (i_prev_poses[pi * 2 + 1] + i_output[pi * 3 + 1] * 2 > resolution - 25 or i_prev_poses[pi * 2 + 1] + i_output[pi * 3 + 1] * 2 < 25 or
                        i_prev_poses[pi * 2] + i_output[pi * 3] * 2 > resolution - 25 or i_prev_poses[pi * 2] + i_output[pi * 3] * 2 < 25):
                    valid = False
                    break
                # find out transformation
                mask = i_rotated_mask_objs[i_sort_idx[pi]]
                points = np.argwhere(mask == 255)
                points = np.expand_dims(points, axis=0)
                M = cv2.getRotationMatrix2D((i_prev_poses[pi * 2 + 1], i_prev_poses[pi * 2]), -i_output[pi * 3 + 2], 1)
                M[0, 2] += i_output[pi * 3 + 1]
                M[1, 2] += i_output[pi * 3]
                new_points = cv2.transform(points, M)
                post_points_pad.append(tuple(np.transpose(points[0] + padding_width)))
                post_new_points_pad.append(tuple(np.transpose(new_points[0] + padding_width)))
            validations.append(valid)
            if valid:
                for pi in range(num_obj[i]):
                    generated_color_image[post_new_points_pad[pi]] = color_image[post_points_pad[pi]]
                    generated_depth_image[post_new_points_pad[pi]] = depth_image[post_points_pad[pi]]
                    if plot:
                        cv2.circle(generated_color_image,
                                   (i_prev_poses[pi * 2] + 48, i_prev_poses[pi * 2 + 1] + 48), 3, (255, 255, 255), -1)
                if plot:
                    cv2.arrowedLine(generated_color_image, (action_start_ori[i][0] + 48, action_start_ori[i][1] + 48),
                                    (action_end_ori[i][0] + 48, action_end_ori[i][1] + 48), (255, 0, 255), 2, tipLength=0.4)
                generated_color_image = imutils.rotate(generated_color_image, angle=-i_rotated_angle)
                generated_depth_image = imutils.rotate(generated_depth_image, angle=-i_rotated_angle)
                generated_color_image = generated_color_image[48:272, 48:272, :]
                generated_depth_image = generated_depth_image[48:272, 48:272]
                generated_color_image = cv2.medianBlur(generated_color_image, 5)
                generated_depth_image = generated_depth_image.astype(np.float32)
                generated_depth_image = cv2.medianBlur(generated_depth_image, 5)

            generated_color_images.append(generated_color_image)
            generated_depth_images.append(generated_depth_image)

        return generated_color_images, generated_depth_images, validations


def sample_actions(color_image, depth_image, mask_objs, plot=False, start_pose=None):
    """
    Sample actions around the objects, from the boundary to the center.
    Assume there is no object in "black"
    Output the rotated image, such that the push action is from left to right
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if plot:
        plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    # find the contour of a single object
    points_on_contour = []
    points = []
    center = []
    binary_objs = []
    for oi in range(len(mask_objs)):
        obj_cnt = cv2.findContours(mask_objs[oi], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        obj_cnt = imutils.grab_contours(obj_cnt)
        obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))[-1]  # the mask r cnn could give bad masks
        # get center
        M = cv2.moments(obj_cnt)
        cX = int(round(M["m10"] / M["m00"]))
        cY = int(round(M["m01"] / M["m00"]))
        center.append([cX, cY])
        if plot:
            cv2.circle(plot_image, (cX, cY), 3, (255, 255, 255), -1)
        # get contour points
        for p in range(0, len(obj_cnt), 15):  # skip 15 points
            x = obj_cnt[p][0][0]
            y = obj_cnt[p][0][1]
            if x == cX or y == cY:
                continue
            diff_x = cX - x
            diff_y = cY - y
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            points_on_contour.append((int(round(x)), int(round(y))))
            points.append((int(round(x - diff_x * 15)), int(round(y - diff_y * 15))))
        # get crop of each object
        temp = np.zeros((resolution_pad, resolution_pad), dtype=np.uint8)
        temp[padding_width:resolution_pad - padding_width, padding_width:resolution_pad - padding_width] = mask_objs[oi]
        crop = temp[cY + 48 - 30:cY + 48 + 30, cX + 48 - 30:cX + 48 + 30]
        assert crop.shape[0] == 60 and crop.shape[1] == 60, (crop.shape)
        binary_objs.append(crop)

    if plot:
        # loop over the contours
        for c in cnts:
            cv2.drawContours(plot_image, [c], -1, (0, 255, 0), 2)

    valid_points = []
    for pi in range(len(points)):
        # out of boundary
        if points[pi][0] < 10 or points[pi][0] > resolution - 10 or points[pi][1] < 10 or points[pi][1] > resolution - 10:
            qualify = False
        # clearance
        elif np.sum(depth_image[points[pi][1] - 6:points[pi][1] + 7, points[pi][0] - 6:points[pi][0] + 7] > DEPTH_MIN) == 0:
            qualify = True
        else:
            qualify = False
        if qualify:
            if plot:
                diff_x = points_on_contour[pi][0] - points[pi][0]
                diff_y = points_on_contour[pi][1] - points[pi][1]
                diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                diff_x /= diff_norm
                diff_y /= diff_norm
                cv2.arrowedLine(plot_image, (int(points[pi][0] - diff_x * 5), int(points[pi][1] - diff_y * 5)), points_on_contour[pi], (250, 0, 250), 2, tipLength=0.4)
            valid_points.append([points[pi], points_on_contour[pi]])
    if start_pose is not None:
        print(valid_points)
        x = (float(start_pose[0]) - workspace_limits[0][0]) / heightmap_resolution
        y = (float(start_pose[1]) - workspace_limits[1][0]) / heightmap_resolution
        start_pose = (int(round(x)), int(round(y)))
        end_pose = (int(round(x + 5 / 0.2)), int(round(y)))
        valid_points = [[start_pose, end_pose]]
        print(valid_points)

    if plot:
        cv2.imwrite('test.png', plot_image)

    # rotate image
    rotated_color_image = []
    rotated_depth_image = []
    rotated_mask_objs = []
    rotated_angle = []
    rotated_center = []
    rotated_action = []
    rotated_binary_objs_image = []
    before_rotated_action = []
    count = 0
    for action in valid_points:
        # padding from 224 to 320
        # color image
        color_image_pad = np.zeros((320, 320, 3), np.uint8)
        color_image_pad[padding_width:resolution_pad - padding_width,
                        padding_width:resolution_pad - padding_width] = color_image
        # depth image
        depth_image_pad = np.zeros((320, 320), np.float32)
        depth_image_pad[padding_width:resolution_pad - padding_width,
                        padding_width:resolution_pad - padding_width] = depth_image

        # compute rotation angle
        right = (1, 0)
        current = (action[1][0] - action[0][0], action[1][1] - action[0][1])
        dot = right[0] * current[0] + right[1] * current[1]      # dot product between [x1, y1] and [x2, y2]
        det = right[0] * current[1] - right[1] * current[0]      # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle)
        rotated_angle.append(angle)
        # rotate images
        rotated_color = imutils.rotate(color_image_pad, angle=angle)
        rotated_depth = imutils.rotate(depth_image_pad, angle=angle)
        rotated_color_image.append(rotated_color)
        rotated_depth_image.append(rotated_depth)
        # rotate cropped object
        if len(binary_objs) == 1:
            # binary_objs_image = np.expand_dims(binary_objs[0], axis=-1)
            binary_objs_image = binary_objs[0]
            rotated_binary_objs = imutils.rotate(binary_objs_image, angle=angle)
            rotated_binary_objs = np.expand_dims(rotated_binary_objs, axis=-1)
        else:
            binary_objs_image = np.stack(binary_objs, axis=-1)
            rotated_binary_objs = imutils.rotate(binary_objs_image, angle=angle)
        M = cv2.getRotationMatrix2D((resolution / 2, resolution / 2), angle, 1)  # rotate by center
        # rotate points
        points = np.array(center)
        points = np.concatenate((points, [action[0]]), axis=0)
        points = np.expand_dims(points, axis=0)
        points = cv2.transform(points, M)[0]
        points_center = points[:len(center)]
        rotated_center.append(points_center)
        rotated_action.append(points[-1])
        rotated_binary_objs_image.append(rotated_binary_objs)
        rotated_mask_obj = []
        for mask in mask_objs:
            rotated_mask_obj.append(imutils.rotate(mask, angle=angle))
        rotated_mask_objs.append(rotated_mask_obj)
        before_rotated_action.append(action[0])

        if plot:
            rotated_image = rotated_depth.copy()
            rotated_image = rotated_image[padding_width:resolution_pad -
                                          padding_width, padding_width:resolution_pad - padding_width]
            rotated_image[rotated_image > DEPTH_MIN] = 255
            rotated_image[rotated_image <= DEPTH_MIN] = 0
            rotated_image = rotated_image.astype(np.uint8)
            for ci in range(len(points_center)):
                cX, cY = rotated_center[-1][ci]
                cv2.circle(rotated_image, (cX, cY), 3, (128), -1)
            x1, y1 = rotated_action[-1]
            cv2.arrowedLine(rotated_image, (x1, y1), (x1 + int((distance / 100) / heightmap_resolution), y1),
                            (128), 2, tipLength=0.4)  # 10 cm away
            cv2.circle(rotated_image, (x1, y1), 2, (200), -1)
            cv2.imwrite(str(count) + 'test_rotated.png', rotated_image)
            count += 1

    return rotated_color_image, rotated_depth_image, rotated_action, rotated_center, rotated_angle, rotated_binary_objs_image, before_rotated_action, rotated_mask_objs


@torch.no_grad()
def from_maskrcnn(model, color_image, device, plot=False):
    """
    Use Mask R-CNN to do instance segmentation and output masks in binary format.
    """
    model.eval()

    image = color_image.copy()
    image = TF.to_tensor(image)
    prediction = model([image.to(device)])[0]

    mask_objs = []
    if plot:
        pred_mask = np.zeros((224, 224), dtype=np.uint8)
        print(prediction['scores'])
    for idx, mask in enumerate(prediction['masks']):
        # TODO, 0.95 and 128 can be tuned
        if prediction['scores'][idx] > 0.95:
            img = mask[0].mul(255).byte().cpu().numpy()
            img = cv2.medianBlur(img, 5)
            img[img > 128] = 255
            img[img <= 128] = 0
            if np.sum(img == 255) < 200:
                continue
            mask_objs.append(img)
            if plot:
                pred_mask[img > 0] = 255 - idx * 50
                cv2.imwrite(str(idx) + 'mask.png', img)
    if plot:
        cv2.imwrite('pred.png', pred_mask)
    print("Mask R-CNN: %d objects detected" % len(mask_objs), prediction['scores'].cpu())
    return mask_objs


if __name__ == "__main__":
    mp.set_start_method('spawn')

    # color_image = cv2.imread("logs_push/push-test/data/color-heightmaps/0003308.color.png")
    # color_image_after = cv2.imread("logs_push/final-test/data/color_heightmaps/0002507.color.png")
    # color_image = cv2.imread("logs/action_test/data/color-heightmaps/000004.0.color.png")
    # color_image = cv2.imread("logs/temp/data/color-heightmaps/000001.0.color.png")
    # color_image = cv2.imread("logs/real-maskrcnn/data/color-heightmaps/000002.0.color.png")
    # color_image = cv2.imread("logs/old/object-detection-data/data/color-heightmaps/000001.0.color.png")
    color_image = cv2.imread("data/real-data/data/color-heightmaps/000000.0.color.png")
    # color_image = cv2.imread("logs/vpg+&pp/p104/data/color-heightmaps/000001.0.color.png")
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # color_image_after = cv2.cvtColor(color_image_after, cv2.COLOR_BGR2RGB)
    # depth_image = cv2.imread("logs/grasp-only/data/depth-heightmaps/000018.0.depth.png", cv2.IMREAD_UNCHANGED)
    # depth_image = cv2.imread("logs/action_test/data/depth-heightmaps/000004.0.depth.png", cv2.IMREAD_UNCHANGED)
    # depth_image = cv2.imread("logs_push/push-test/data/depth-heightmaps/0003308.depth.png", cv2.IMREAD_UNCHANGED)
    # depth_image = cv2.imread("logs/real-maskrcnn/data/depth-heightmaps/000002.0.depth.png", cv2.IMREAD_UNCHANGED)
    # depth_image = cv2.imread("logs/old/object-detection-data/data/depth-heightmaps/000001.0.depth.png", cv2.IMREAD_UNCHANGED)
    depth_image = cv2.imread("data/real-data/data/depth-heightmaps/000000.0.depth.png", cv2.IMREAD_UNCHANGED)
    # depth_image = cv2.imread("logs/vpg+&pp/p104/data/depth-heightmaps/000001.0.depth.png", cv2.IMREAD_UNCHANGED)
    depth_image = depth_image.astype(np.float32) / 100000


    # with open('logs_push/final-test/data/actions/0002502.action.txt', 'r') as file:
    #     filedata = file.read()
    #     x, y = filedata.split(' ')
    # start_pose = [x, y]

    # cv2.imwrite('predicttruth.png', color_image_after)

    # check diff of color image and depth image
    # gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.medianBlur(gray, 5)
    # gray = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    # depth_image[depth_image <= DEPTH_MIN] = 0
    # depth_image[depth_image > DEPTH_MIN] = 255
    # # depth_image = depth_image.astype(np.uint8)
    # cv2.imshow('color', gray)
    # cv2.imwrite('blackwhite', gray)
    # diff = depth_image - gray
    # diff[diff < 0] = 128
    # cv2.imshow('diff', diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # predictor = Predictor("logs_push/final/data/5cm/20push_prediction_model_test_5.pth")
    predictor = Predictor("logs_push/15push_prediction_model_test_5.pth")
    # predictor = Predictor("logs_push/old/new/data/push_prediction_model_test_5.pth")
    trainer = Trainer('reinforcement', 0, 0, True, True,
                      'logs/paper result/grasp-only/models/snapshot-000400.reinforcement.pth', False)
    model = get_model_instance_segmentation(2)
    # model.load_state_dict(torch.load("data/real-data/data/maskrcnn.pth"))
    model.load_state_dict(torch.load("logs/maskrcnn.pth"))
    model = model.to(device)

    mask_objs = from_maskrcnn(model, color_image, device, True)
    rotated_color_image, rotated_depth_image, rotated_action, rotated_center, rotated_angle, rotated_binary_objs, before_rotated_action, rotated_mask_objs = sample_actions(
        color_image, depth_image, mask_objs, True)

    generated_color_images, generated_depth_images, validations = predictor.forward(
        rotated_color_image, rotated_depth_image, rotated_action, rotated_center, rotated_angle, rotated_binary_objs, rotated_mask_objs, False)
    for idx, img in enumerate(generated_color_images):
        overlay = color_image
        # added_image = cv2.addWeighted(generated_color_images[idx], 0.8, overlay, 0.4, 0)
        added_image = generated_color_images[idx].copy()
        img = cv2.cvtColor(added_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(idx) + 'predict.png', img)
        img = generated_depth_images[idx]
        img[img <= DEPTH_MIN] = 0
        img[img > DEPTH_MIN] = 255
        cv2.imwrite(str(idx) + 'predictgray.png', img)

    generated_color_images.append(color_image)
    generated_depth_images.append(depth_image)
    for idx, img in enumerate(generated_color_images):

        if idx + 1 == len(generated_color_images) or validations[idx]:
            _, grasp_predictions = trainer.forward(
                generated_color_images[idx], generated_depth_images[idx], is_volatile=True)
            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            predicted_value = np.max(grasp_predictions)
            grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, generated_color_images[idx], best_pix_ind)
            cv2.imwrite(str(idx) + 'visualization.grasp.png', grasp_pred_vis)
            predicted_values = np.sum(np.sort(grasp_predictions.flatten())[:])
            print(idx, predicted_value, predicted_values)
        else:
            print('invalid')
