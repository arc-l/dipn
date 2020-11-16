import torch
from torchvision import transforms as T
from push_net import PushPredictionNet
from dataset import PushPredictionMultiDataset, ClusterRandomSampler
from torch.utils.data.sampler import RandomSampler
import utils
import argparse
import time
import os
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
import signal
import sys
from constants import color_mean, color_std, total_obj


def signal_handler(signal, frame):
    global writer
    writer.close()
    sys.exit(0)

def get_data_loader(dataset_root, batch_size, distance, shuffle=True, test=False, cutoff=None):
    # use our dataset and defined transformations
    dataset = PushPredictionMultiDataset(dataset_root, distance, False, cutoff)

    # split the dataset in train and test set
    if test:
        data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True)
        data_loader_test = None
    else:
        # could change 5 to a larger value
        dataset_root_test = dataset_root[:-5] + '-test/data'
        dataset_test = PushPredictionMultiDataset(dataset_root_test, distance, False)

        # define training and validation data loaders
        sampler_train = ClusterRandomSampler(dataset, batch_size, True)
        sampler_test = ClusterRandomSampler(dataset_test, int(batch_size / 2), False)
        data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler_train, shuffle=False, num_workers=8, drop_last=False)
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=int(batch_size / 2), sampler=sampler_test, shuffle=False, num_workers=8, drop_last=False)

    return data_loader_train, data_loader_test

def train_one_epoch(args, model, criterion, optimizer, data_loader, device, epoch, print_freq, resume=False):
    global n_iter, writer
    '''
    https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    '''
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.10f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and not resume:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx in metric_logger.log_every(data_loader, print_freq, header):
        used_binary_img = used_binary_img.to(device, non_blocking=True, dtype=torch.float)
        prev_poses = prev_poses.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        delta = delta.to(device, non_blocking=True)
        binary_objs_total = binary_objs_total.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
        target = delta        

        loss = criterion(output, target)
        writer.add_scalar('Loss/train', loss.cpu(), n_iter)
        writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], n_iter)
        n_iter += 1
        loss.backward()

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(args, model, criterion, data_loader, device, print_freq=100):
    global n_iter, writer, n_eva_iter

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx in metric_logger.log_every(data_loader, print_freq, header):
        prev_poses = prev_poses.to(device, non_blocking=True)
        used_binary_img = used_binary_img.to(device, non_blocking=True, dtype=torch.float)
        binary_objs_total = binary_objs_total.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        delta = delta.to(device, non_blocking=True)

        output = model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
        target = delta       

        loss = criterion(output, target)
        writer.add_scalar('Loss/test', loss.cpu(), n_eva_iter)
        n_eva_iter += 1

        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()


def main(args):
    data_loader, data_loader_test = get_data_loader(args.dataset_root, args.batch_size, args.distance, cutoff=args.cutoff)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.resume:

        model = PushPredictionNet()
        if args.snapshot_file:
            state = torch.load(os.path.abspath(args.snapshot_file))
        else:
            state = torch.load(os.path.join(args.dataset_root, "push_prediction_model.pth"))
        model.load_state_dict(state)
    else:
        model = PushPredictionNet()

    model = model.to(device)
    print("max_memory_allocated (MB):", torch.cuda.max_memory_allocated() / 2**20)
    print("memory_allocated (MB):", torch.cuda.memory_allocated() / 2**20)

    criterion = torch.nn.SmoothL1Loss()
    criterion_test = torch.nn.SmoothL1Loss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=2e-4)

    # and a learning rate scheduler which decreases the learning rate by 10x every 1 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs/3, gamma=0.1)

    for epoch in range(args.epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(args, model, criterion, optimizer, data_loader, device, epoch, print_freq=20, resume=args.resume)
        # update the learning rate
        lr_scheduler.step()
        
        if args.snapshot_file:
            torch.save(model.state_dict(), os.path.abspath(args.snapshot_file))
        else:
            torch.save(model.state_dict(), os.path.join(args.dataset_root, str(args.cutoff) + "_push_prediction_model.pth"))
            # torch.save(model.state_dict(), os.path.join(args.dataset_root, str(args.cutoff) + "push_prediction_model_test_5_only_front.pth"))

        # evaluate on the test dataset
        evaluate(args, model, criterion_test, data_loader_test, device, print_freq=8)


@torch.no_grad()
def test(args):
    import torchvision
    import matplotlib.pyplot as plt
    from PIL import Image, ImageStat
    import math
    from constants import total_obj, color_mean, color_std, colors_upper, colors_lower

    torch.manual_seed(1)

    inv_normalize = T.Normalize(mean=[-color_mean[0] / color_std[0], -color_mean[1] / color_std[1], -color_mean[2] / color_std[2]], 
                                std=[1 / color_std[0], 1 / color_std[1], 1 / color_std[2]])


    data_loader, data_loader_test = get_data_loader(args.dataset_root, 1, args.distance, test=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = PushPredictionNet()
    if args.snapshot_file:
        model.load_state_dict(torch.load(os.path.abspath(args.snapshot_file)))
    else:
        model.load_state_dict(torch.load(os.path.join(args.dataset_root, "push_prediction_model_test.pth")))
    model = model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss(reduction='none')

    ite = iter(data_loader)
    images = []
    refs = []
    for i in range(len(data_loader.dataset)):
        prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx= next(ite)
        prev_color_img = prev_color_img.to(device, non_blocking=True)
        prev_depth_img = prev_depth_img.to(device, non_blocking=True)
        next_color_img = next_color_img.to(device, non_blocking=True)
        next_depth_img = next_depth_img.to(device, non_blocking=True)
        used_binary_img = used_binary_img.to(device, non_blocking=True, dtype=torch.float)
        prev_poses = prev_poses.to(device, non_blocking=True)
        next_poses = next_poses.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        delta = delta.to(device, non_blocking=True)
        action_start_ori = action_start_ori.to(device, non_blocking=True)
        action_end_ori = action_end_ori.to(device, non_blocking=True)
        binary_objs_total = binary_objs_total.to(device, non_blocking=True)
        sort_idx = sort_idx[0].numpy()
        print('sort:', sort_idx)
    
        output = model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
        target = delta
        loss = criterion(output, target)
        output = output[0].cpu().numpy()
        target = target[0].cpu().numpy()
        # output = target
        output_xy = []
        output_a = []
        for num_idx in range(num_obj):
            output_xy.append([output[num_idx * 3], output[num_idx * 3 + 1]])
            output_a.append(output[num_idx * 3 + 2])
            # no move
            # output_xy.append([0, 0])
            # output_a.append(0)
            # move in the direction as the action
            # output_xy.append([args.distance / 0.2, 0])
            # output_a.append(0)
        print(i)
        print(prev_ref[0])
        print(next_ref[0])
        np.set_printoptions(precision=3, suppress=True)
        print('output', output)
        print('target', target)
        print('action', action_start_ori.cpu().numpy())
        print('loss', loss.cpu().numpy())
        loss = loss.cpu().numpy()[0]

        next_color_img = inv_normalize(next_color_img[0])
        next_color_img = next_color_img.cpu().permute(1, 2, 0).numpy()
        imdepth = next_color_img
        imdepth[imdepth < 0] = 0
        imdepth[imdepth > 0] = 255
        imdepth = imdepth.astype(np.uint8)
        imdepth = cv2.cvtColor(imdepth, cv2.COLOR_RGB2BGR)

        prev_color_img = inv_normalize(prev_color_img[0])
        prev_color_img = prev_color_img.cpu().permute(1, 2, 0).numpy()
        imgcolor = prev_color_img
        imgcolor[imgcolor < 0] = 0
        imgcolor *= 255
        imgcolor = imgcolor.astype(np.uint8)
        imgcolor = cv2.GaussianBlur(imgcolor, (5, 5), 0)
        imgcolor = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2HSV)

        prev_poses = prev_poses[0].cpu().numpy()
        next_poses = next_poses[0].cpu().numpy()
        action_start_ori = action_start_ori[0].cpu().numpy()
        action_end_ori = action_end_ori[0].cpu().numpy()
        action_start_ori_tile = np.tile(action_start_ori, num_obj[0])
        action = action[0].cpu().numpy()
        action_start_tile = np.tile(action[:2], num_obj[0])
        prev_poses += action_start_ori_tile
        prev_poses -= action_start_tile
        next_poses += action_start_ori_tile
        next_poses -= action_start_tile
        print('prev poses', prev_poses)
        print('next poses', next_poses)

        for ci in range(num_obj):
            color = cv2.inRange(imgcolor, colors_lower[sort_idx[ci]], colors_upper[sort_idx[ci]])
            contours, _ = cv2.findContours(color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            found = False
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    contours = contour
                    found = True
                    break
            if not found: continue
            cv2.drawContours(imdepth, [contours], -1, (255 / int(num_obj) * (ci + 1), 255, 255 - 255 / int(num_obj) * (ci + 1)), 1)
            cnt_rotated = rotate_contour(contours, -output_a[ci])
            cnt_rotated_translated = cnt_rotated + output_xy[ci]
            cnt_rotated_translated = np.rint(cnt_rotated_translated).astype(np.int32)
            cv2.drawContours(imdepth, [cnt_rotated_translated], -1, (255 / int(num_obj) * (ci + 1), 150, 255 - 255 / int(num_obj) * (ci + 1)), 2)
        
        for pi in range(num_obj):
            cv2.circle(imdepth, (int(round(prev_poses[pi * 2])), int(round(prev_poses[pi * 2 + 1]))), 2, (255, 0, 255), -1)
            cv2.circle(imdepth, (int(round(next_poses[pi * 2])), int(round(next_poses[pi * 2 + 1]))), 2, (255, 255, 0), -1)
        
        # action
        cv2.circle(imdepth, (int(round(action_start_ori[0])), int(round(action_start_ori[1]))), 5, (255, 0, 0), -1)
        cv2.circle(imdepth, (int(round(action_end_ori[0])), int(round(action_end_ori[1]))), 5, (0, 0, 255), -1)

        # if math.sqrt(loss[0]) > 5 or math.sqrt(loss[1]) > 5 or math.sqrt(loss[3]) > 5 or math.sqrt(loss[4]) > 5 or math.sqrt(loss[2]) > 5 or math.sqrt(loss[5]) > 5:
        images.append(cv2.cvtColor(imdepth, cv2.COLOR_BGR2RGB))
        refs.append(prev_ref[0])
        if len(images) == 28:
            for i in range(len(images)):
                plt.subplot(math.ceil(len(images) / 7), 7, i+1),plt.imshow(images[i], 'gray')
                plt.title(refs[i][:7])
                plt.xticks([]),plt.yticks([])
            # plt.show()
            plt.savefig('test.png', dpi=400)
            input_str = input('One more?')
            if input_str == 'y':
                images = []
                refs = []
            else:
                break

@torch.no_grad()
def plot(args):
    import torchvision
    import matplotlib.pyplot as plt
    from PIL import Image, ImageStat
    import math
    from constants import total_obj, color_mean, color_std, colors_upper, colors_lower

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(1)

    inv_normalize = T.Normalize(mean=[-color_mean[0] / color_std[0], -color_mean[1] / color_std[1], -color_mean[2] / color_std[2]], 
                                std=[1 / color_std[0], 1 / color_std[1], 1 / color_std[2]])


    data_loader, data_loader_test = get_data_loader(args.dataset_root, 1, args.distance, False, test=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = PushPredictionNet()
    if args.snapshot_file:
        model.load_state_dict(torch.load(os.path.abspath(args.snapshot_file)))
    else:
        model.load_state_dict(torch.load(os.path.join(args.dataset_root, "push_prediction_model_test.pth")))
    model = model.to(device)
    model.eval()
    criterion = torch.nn.SmoothL1Loss(reduction='none')

    total_num = len(data_loader.dataset)
    print(total_num)
    ite = iter(data_loader)
    images = []
    refs = []
    for i in range(total_num):
        prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx= next(ite)
        if i % 5 == 0 or args.distance != 5:
            prev_color_img = prev_color_img.to(device, non_blocking=True)
            prev_depth_img = prev_depth_img.to(device, non_blocking=True)
            next_color_img = next_color_img.to(device, non_blocking=True)
            next_depth_img = next_depth_img.to(device, non_blocking=True)
            used_binary_img = used_binary_img.to(device, non_blocking=True, dtype=torch.float)
            prev_poses = prev_poses.to(device, non_blocking=True)
            next_poses = next_poses.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            delta = delta.to(device, non_blocking=True)
            action_start_ori = action_start_ori.to(device, non_blocking=True)
            action_end_ori = action_end_ori.to(device, non_blocking=True)
            binary_objs_total = binary_objs_total.to(device, non_blocking=True)
            sort_idx = sort_idx[0].numpy()
            print('sort:', sort_idx)
        
            output = model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])
            target = delta
            loss = criterion(output, target)
            output = output[0].cpu().numpy()
            target = target[0].cpu().numpy()
            output_xy = []
            output_a = []
            for num_idx in range(num_obj):
                output_xy.append([output[num_idx * 3], output[num_idx * 3 + 1]])
                output_a.append(output[num_idx * 3 + 2])
            print(i)
            print(prev_ref[0])
            print(next_ref[0])
            np.set_printoptions(precision=3, suppress=True)
            print('output', output)
            print('target', target)
            print('action', action_start_ori.cpu().numpy())
            print('loss', loss.cpu().numpy())
            loss = loss.cpu().numpy()[0]

            # background
            next_color_img = inv_normalize(next_color_img[0])
            next_color_img = next_color_img.cpu().permute(1, 2, 0).numpy()
            imnext = next_color_img
            imnext[imnext < 0] = 0
            imnext *= 255
            imnext = imnext.astype(np.uint8)
            imnext = cv2.cvtColor(imnext, cv2.COLOR_RGB2BGR)

            prev_color_img = inv_normalize(prev_color_img[0])
            prev_color_img = prev_color_img.cpu().permute(1, 2, 0).numpy()
            imgcolor = prev_color_img
            imgcolor[imgcolor < 0] = 0
            imgcolor *= 255
            imgcolor = imgcolor.astype(np.uint8)
            imgcolor = cv2.GaussianBlur(imgcolor, (5, 5), 0)
            imgcolorhsv = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2HSV)

            prev_poses = prev_poses[0].cpu().numpy()
            next_poses = next_poses[0].cpu().numpy()
            action_start_ori = action_start_ori[0].cpu().numpy()
            action_end_ori = action_end_ori[0].cpu().numpy()
            action_start_ori_tile = np.tile(action_start_ori, num_obj[0])
            action = action[0].cpu().numpy()
            action_start_tile = np.tile(action[:2], num_obj[0])
            prev_poses += action_start_ori_tile
            prev_poses -= action_start_tile
            next_poses += action_start_ori_tile
            next_poses -= action_start_tile
            print('prev poses', prev_poses)
            print('next poses', next_poses)

            newimg = np.zeros_like(imnext)
            for ci in range(num_obj):
                color = cv2.inRange(imgcolorhsv, colors_lower[sort_idx[ci]], colors_upper[sort_idx[ci]])

                if np.sum(color == 255) > 100:
                    points = np.argwhere(color == 255)
                    points = np.expand_dims(points, axis=0)
                    M = cv2.getRotationMatrix2D((prev_poses[ci * 2 + 1], prev_poses[ci * 2]), -output[ci * 3 + 2], 1)
                    M[0, 2] += output[ci * 3 + 1]
                    M[1, 2] += output[ci * 3]
                    new_points = cv2.transform(points, M)
                    newimg[tuple(np.transpose(new_points))] = imgcolor[tuple(np.transpose(points))]
                            
            # action
            cv2.arrowedLine(imnext, (action_start_ori[0], action_start_ori[1]),
                                    (action_end_ori[0], action_end_ori[1]), (255, 255, 255), 2, tipLength=0.4)
            cv2.arrowedLine(imgcolor, (action_start_ori[0], action_start_ori[1]),
                                    (action_end_ori[0], action_end_ori[1]), (255, 255, 255), 2, tipLength=0.4)

            newimg = cv2.medianBlur(newimg, 5)
            newimg = cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR)
            newimg = cv2.addWeighted(newimg, 0.4, imnext, 0.6, 0)
            images.append(imgcolor)
            refs.append(prev_ref[0][3:7])
            images.append(cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB))
            refs.append("prediction of " + str(prev_ref[0][3:7]))
            if len(images) == 32:
                for i in range(len(images)):
                    cv2.imwrite('figures/push-prediction-plot/' + refs[i] + '.png', cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
                    plt.subplot(math.ceil(len(images) / 8), 8, i+1),plt.imshow(images[i], 'gray')
                    plt.title(refs[i])
                    plt.xticks([]),plt.yticks([])
                plt.show()
                # plt.savefig('test.png', dpi=400)
                input_str = input('One more?')
                if input_str == 'y':
                    images = []
                    refs = []
                else:
                    break


@torch.no_grad()
def verify_dataset(args):
    import torchvision
    import matplotlib.pyplot as plt
    from PIL import Image, ImageStat
    import math
    from constants import color_mean, color_std, depth_mean, depth_std

    torch.manual_seed(1244)

    inv_normalize_color = T.Normalize(mean=[-color_mean[0] / color_std[0], -color_mean[1] / color_std[1], -color_mean[2] / color_std[2]], 
                                std=[1 / color_std[0], 1 / color_std[1], 1 / color_std[2]])
    inv_normalize_depth = T.Normalize(mean=[-depth_mean[0] / depth_std[0]], 
                                std=[1 / depth_std[0]])


    data_loader, data_loader_test = get_data_loader(args.dataset_root, 1, args.distance, test=True)

    ite = iter(data_loader)
    images = []
    refs = []
    for i in range(len(data_loader.dataset)):
        prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx = next(ite)

        binary_objs_total = binary_objs_total[0].numpy().astype(np.uint8)
        num_obj = len(binary_objs_total)
        for i in range(num_obj):
            temp = binary_objs_total[i]
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            temp *= 255
            temp = temp.astype(np.uint8)
            cv2.imshow(str(i), temp)

        np.set_printoptions(precision=3, suppress=True)
        prev_poses = prev_poses[0].numpy().astype(int)
        action = action[0].numpy().astype(int)
        action_start_tile = np.tile(action[:2], num_obj)
        print('prev poses', prev_poses)

        img = inv_normalize_color(prev_color_img[0])
        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img *= 255
        img = img.astype(np.uint8)
        cv2.imshow('prev color', img)

        img = inv_normalize_depth(prev_depth_img[0])
        img = img.permute(1, 2, 0).numpy()
        cv2.imshow('prev depth', img)

        img = used_binary_img[0, 0].numpy().astype(int)
        img *= 255
        img = img.astype(np.uint8)
        for pi in range(num_obj):
            cv2.circle(img, (prev_poses[pi * 2], prev_poses[pi * 2 + 1]), 2, (120, 102, 255), -1)
        cv2.imshow('prev binary', img)

        img = used_binary_img[0][1].numpy()
        img *= 255
        img = img.astype(np.uint8)
        cv2.circle(img, (action[0], action[1]), 2, (120, 102, 255), -1)
        cv2.circle(img, (action[2], action[3]), 2, (120, 102, 255), -1)
        cv2.imshow('action', img)

        img = inv_normalize_color(next_color_img[0])
        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img *= 255
        img = img.astype(np.uint8)
        cv2.imshow('next color', img)

        img = inv_normalize_depth(next_depth_img[0])
        img = img.permute(1, 2, 0).numpy()
        cv2.imshow('next depth', img)
        
        action_start_ori = action_start_ori[0].numpy().astype(int)
        action_end_ori = action_end_ori[0].numpy().astype(int)
        action_start_ori_tile = np.tile(action_start_ori, num_obj)

        prev_imdepth = prev_depth_img[0].cpu().permute(1, 2, 0).numpy()
        prev_imdepth[prev_imdepth <= 0] = 0
        prev_imdepth[prev_imdepth > 0] = 255
        prev_imdepth = np.repeat(prev_imdepth, 3, axis=2)
        prev_imdepth = prev_imdepth.astype(np.uint8)
        cv2.circle(prev_imdepth, (action_start_ori[0], action_start_ori[1]), 5, (255, 0, 0), -1)
        cv2.circle(prev_imdepth, (action_end_ori[0], action_end_ori[1]), 5, (0, 0, 255), -1)
        prev_poses += action_start_ori_tile
        prev_poses -= action_start_tile
        for pi in range(num_obj):
            cv2.circle(prev_imdepth, (prev_poses[pi * 2], prev_poses[pi * 2 + 1]), 2, (255, 0, 255), -1)
        print('prev poses', prev_poses)

        next_imdepth = next_depth_img[0].cpu().permute(1, 2, 0).numpy()
        next_imdepth[next_imdepth <= 0] = 0
        next_imdepth[next_imdepth > 0] = 255
        next_imdepth = np.repeat(next_imdepth, 3, axis=2)
        next_imdepth = next_imdepth.astype(np.uint8)
        cv2.circle(next_imdepth, (action_start_ori[0], action_start_ori[1]), 5, (255, 0, 0), -1)
        cv2.circle(next_imdepth, (action_end_ori[0], action_end_ori[1]), 5, (0, 0, 255), -1)
        next_poses = next_poses[0].numpy().astype(int)
        next_poses += action_start_ori_tile
        next_poses -= action_start_tile
        for pi in range(num_obj):
            cv2.circle(next_imdepth, (next_poses[pi * 2], next_poses[pi * 2 + 1]), 2, (255, 255, 0), -1)
        print('next poses', next_poses)

        delta = delta[0].numpy()
        print('delta', delta)
        
        cv2.imshow('prev imdepth', prev_imdepth)
        cv2.imshow('next imdepth', next_imdepth)

        k = cv2.waitKey(0) 
        cv2.destroyAllWindows()
        if k == ord('q'):  # ESC
            break

@torch.no_grad()
def symmetric_diff(args):
    import torchvision
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from PIL import Image, ImageStat
    from torchvision.transforms import functional as TF
    from constants import total_obj, workspace_limits, heightmap_resolution, color_mean, color_std, colors_upper, colors_lower
    import math

    inv_normalize = T.Normalize(mean=[-color_mean[0] / color_std[0], -color_mean[1] / color_std[1], -color_mean[2] / color_std[2]], 
                            std=[1 / color_std[0], 1 / color_std[1], 1 / color_std[2]])

    data_loader, data_loader_test = get_data_loader(args.dataset_root, 1, args.distance, False, test=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = PushPredictionNet()
    if args.snapshot_file:
        model.load_state_dict(torch.load(os.path.abspath(args.snapshot_file)))
    else:
        model.load_state_dict(torch.load(os.path.join(args.dataset_root, "push_prediction_model_test_sort.pth")))
    model = model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss(reduction='none')

    total_symmetric_difference = []
    total_area = []
    total_num = len(data_loader.dataset)
    print(total_num)
    ite = iter(data_loader)
    for i in range(int(total_num)):
        prev_color_img, prev_depth_img, next_color_img, next_depth_img, used_binary_img, prev_poses, next_poses, action, delta, prev_ref, next_ref, action_start_ori, action_end_ori, binary_objs_total, num_obj, sort_idx = next(ite)
        # if i % 5 == 0 or args.distance != 5:
        if True:
            prev_color_img = prev_color_img.to(device, non_blocking=True)
            prev_depth_img = prev_depth_img.to(device, non_blocking=True)
            next_color_img = next_color_img.to(device, non_blocking=True)
            next_depth_img = next_depth_img.to(device, non_blocking=True)
            used_binary_img = used_binary_img.to(device, non_blocking=True, dtype=torch.float)
            prev_poses = prev_poses.to(device, non_blocking=True)
            next_poses = next_poses.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            delta = delta.to(device, non_blocking=True)
            action_start_ori = action_start_ori.to(device, non_blocking=True)
            action_end_ori = action_end_ori.to(device, non_blocking=True)
            binary_objs_total = binary_objs_total.to(device, non_blocking=True)

            output = model(prev_poses, action, used_binary_img, binary_objs_total, num_obj[0])    # output = model(action, prev_poses)
            target = delta
            loss = criterion(output, target)
            output = output[0].cpu().numpy()
            target = target[0].cpu().numpy()
            num_obj = num_obj[0].cpu().item()
            output_xy = []
            output_a = []
            for num_idx in range(num_obj):
                output_xy.append([output[num_idx * 3], output[num_idx * 3 + 1]])
                output_a.append(output[num_idx * 3 + 2])
                # no move
                # output_xy.append([0, 0])
                # output_a.append(0)
                # move in the direction as the action
                # if num_idx == 0:
                #     output_xy.append([(args.distance / 0.2), 0])
                #     output_a.append(0)
                # else:
                #     output_xy.append([0, 0])
                #     output_a.append(0)
            print(i)
            print(prev_ref[0])
            sort_idx = sort_idx[0].numpy()
            print('sort:', sort_idx)
            # print('output', output_x_y1.numpy(), output_a1.numpy(), output_x_y2.numpy(), output_a2.numpy())
            # print('target', target.numpy())
            # print('action', action_start_ori.cpu().numpy())
            # print('loss', loss.cpu().numpy())

            # ===== symmetric difference =====
            prev_poses = prev_poses[0].cpu().numpy().astype(int)
            action_start_ori = action_start_ori[0].cpu().numpy().astype(int)
            action_start_ori_tile = np.tile(action_start_ori, num_obj)
            action = action[0].cpu().numpy().astype(int)
            action_start_tile = np.tile(action[:2], num_obj)
            prev_poses += action_start_ori_tile
            prev_poses -= action_start_tile
            next_img = next_depth_img[0].cpu().permute(1, 2, 0).squeeze().numpy()
            pred_img_colors = [np.zeros((320, 320), dtype=np.uint8) for i in range(num_obj)]
            prev_color_img = inv_normalize(prev_color_img[0])
            prev_color_img = prev_color_img.cpu().permute(1, 2, 0).numpy()
            imgcolor = prev_color_img
            imgcolor *= 255
            imgcolor = imgcolor.astype(np.uint8)
            imgcolor = cv2.medianBlur(imgcolor, 5)
            before_push_img = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2BGR)
            imgcolor = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2HSV)

            # prediction
            for ci in range(num_obj):
                color = cv2.inRange(imgcolor, colors_lower[sort_idx[ci]], colors_upper[sort_idx[ci]])
                points = np.argwhere(color == 255)
                points = np.expand_dims(points, axis=0)
                M = cv2.getRotationMatrix2D((prev_poses[ci * 2 + 1], prev_poses[ci * 2]), -output_a[ci], 1)
                M[0, 2] += output_xy[ci][1]
                M[1, 2] += output_xy[ci][0]
                points = cv2.transform(points, M)
                points[0, :, 0] += 48
                points[0, :, 1] += 48
                pred_img_colors[ci][tuple(np.transpose(points[0]))] = 255
                pred_img_colors[ci] = pred_img_colors[ci][48:(320-48),48:(320-48)]
                pred_img_colors[ci] = cv2.medianBlur(pred_img_colors[ci], 5)

            # ground truth
            next_color_img = inv_normalize(next_color_img[0])
            next_color_img = next_color_img.cpu().permute(1, 2, 0).numpy()
            next_img_color = next_color_img
            next_img_color[next_img_color < 0] = 0
            next_img_color *= 255
            next_img_color = next_img_color.astype(np.uint8)
            imgcolor = cv2.cvtColor(next_img_color, cv2.COLOR_RGB2HSV)
            next_img_colors = []
            for ci in range(num_obj):
                next_img_color = cv2.inRange(imgcolor, colors_lower[sort_idx[ci]], colors_upper[sort_idx[ci]])
                next_img_colors.append(next_img_color)
                total_area.append(np.sum(next_img_color == 255))

            # intersection
            for ci in range(num_obj):
                intersection_color = np.zeros_like(next_img)
                intersection_color[np.logical_and(pred_img_colors[ci] == 255, next_img_colors[ci] == 255)] = 255
                union_color = np.zeros_like(next_img)
                union_color[np.logical_or(pred_img_colors[ci] == 255, next_img_colors[ci] == 255)] = 255
                diff_color = union_color - intersection_color
                total_symmetric_difference.append(np.sum(diff_color == 255))

    print(np.average(total_area))
    print(np.std(total_area))

    diff_union = np.array(total_symmetric_difference) / np.array(total_area)
    print(np.average(diff_union))
    print(np.std(diff_union))
    np.savetxt('test.txt', diff_union)  

    plt.hist(diff_union, weights=np.ones(len(diff_union)) / len(diff_union), range=(0, 2))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.show()
    plt.savefig('test.png')
        

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(round(M['m10']/M['m00']))
    cy = int(round(M['m01']/M['m00']))

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    # thetas = thetas + angle
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]

    return cnt_rotated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train foreground')

    parser.add_argument('--dataset_root', dest='dataset_root', action='store', help='Enter the path to the dataset')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store', help='Enter the path to the pretrained model')
    parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=50, help='Enter the epoch for training')
    parser.add_argument('--batch_size', dest='batch_size', action='store', type=int, default=16, help='Enter the batchsize for training and testing')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='Testing and visualizing')
    parser.add_argument('--verify', dest='verify', action='store_true', default=False, help='')
    parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='')
    parser.add_argument('--symmetric_diff', dest='symmetric_diff', action='store_true', default=False, help='symmetric_diff')
    parser.add_argument('--test_stream', dest='test_stream', action='store_true', default=False, help='Testing and visualizing')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False ,help='Enter the path to the dataset')
    parser.add_argument('--lr', dest='lr', action='store', type=float, default=1e-3, help='Enter the learning rate')
    parser.add_argument('--distance', dest='distance', action='store', type=int, default=5, help='the distance of one push')
    parser.add_argument('--cutoff', dest='cutoff', action='store', type=int, default=15, help='the cutoff of push')

    args = parser.parse_args()

    # writer = SummaryWriter()
    signal.signal(signal.SIGINT, signal_handler)

    if args.verify:
        verify_dataset(args)
    elif args.test_stream:
        test_stream(args)
    elif args.test:
        test(args)
    elif args.symmetric_diff:
        symmetric_diff(args)
    elif args.plot:
        plot(args)
    else:
        writer = SummaryWriter()
        n_iter = 0
        n_eva_iter = 0
        main(args)    
        writer.close()