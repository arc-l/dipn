import torch
from torchvision import transforms as T
from models import reinforcement_net
from dataset import ForegroundDataset
import utils
import argparse
import time
import os
from constants import PUSH_Q, GRASP_Q


def get_data_loader(dataset_root, batch_size, fine_tuning_num=None):
    # use our dataset and defined transformations
    dataset = ForegroundDataset(dataset_root, 16, fine_tuning_num)
    # dataset_test = ForegroundDataset(dataset_root, 16)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    start_point = 5
    dataset = torch.utils.data.Subset(dataset, indices[start_point:])
    dataset_test = torch.utils.data.Subset(dataset, indices[:start_point])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)

    return data_loader, data_loader_test


def train_one_epoch(model, criterion_push, criterion_grasp, optimizer,
                    data_loader, device, epoch, print_freq, resume=False):
    '''
    https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    '''
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and not resume:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for color_images, depth_images, push_targets, grasp_targets in metric_logger.log_every(
            data_loader, print_freq, header):
        color_images = color_images.to(device)
        depth_images = depth_images.to(device)
        push_targets = push_targets.to(device)
        grasp_targets = grasp_targets.to(device)

        optimizer.zero_grad()

        output_probs = model(color_images, depth_images)

        weights = torch.ones(grasp_targets.shape)
        # if it doesn't converge, just restart, expecting the loss to below 60. it should below 100 very soon
        weights[grasp_targets > 0] = 2

        loss1 = criterion_push(output_probs[0], push_targets)
        loss1 = loss1.sum() / push_targets.size(0)
        loss1.backward()
        loss2 = criterion_grasp(output_probs[1], grasp_targets) * weights.cuda()
        loss2 = loss2.sum() / grasp_targets.size(0)
        loss2.backward()
        losses = loss1 + loss2

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses.cpu())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def main(args):
    data_loader, data_loader_test = get_data_loader(args.dataset_root, args.batch_size, args.fine_tuning_num)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = reinforcement_net(True, True)  # TODO: remove use_cuda in model, replace with device
    if args.resume:
        # model.load_state_dict(torch.load('data/pre_train/foreground_model.pth'))
        model.load_state_dict(torch.load(os.path.join(args.dataset_root, "foreground_model.pth")))

    criterion_push = torch.nn.SmoothL1Loss(reduction='none')
    criterion_grasp = torch.nn.SmoothL1Loss(reduction='none')
    # criterion_push = torch.nn.BCEWithLogitsLoss()
    # criterion_grasp = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=2e-5)
    # optimizer = torch.optim.SGD(params, lr=1e-4, momentum=0.9, weight_decay=2e-5)

    # and a learning rate scheduler which decreases the learning rate by 10x every 1 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    # for large dataset
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.5)
    # for small dataset, expect ~ 50 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(args.epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model,
            criterion_push,
            criterion_grasp,
            optimizer,
            data_loader,
            device,
            epoch,
            print_freq=20,
            resume=args.resume)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, criterion, data_loader_test, device=device)

        torch.save(model.state_dict(), os.path.join(args.dataset_root, "foreground_model.pth"))


@torch.no_grad()
def test():
    import torchvision
    import matplotlib.pyplot as plt
    from PIL import Image, ImageStat

    torch.manual_seed(2)

    # data_loader, data_loader_test = get_data_loader('data/pre_train/', 1)
    data_loader, data_loader_test = get_data_loader('logs/real-maskrcnn/data', 1)
    # data_loader, data_loader_test = get_data_loader('logs/final-pretrain/data', 1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = reinforcement_net(True, True) 
    # model.load_state_dict(torch.load('data/pre_train/foreground_model.pth'))
    model.load_state_dict(torch.load('logs/random-pretrain/data/foreground_model.pth'))
    # model.load_state_dict(torch.load('logs/real-maskrcnn/data/foreground_model.pth'))
    # model.load_state_dict(torch.load('logs_push/final/data/foreground_model.pth'))
    model.eval().to(device)
    sig = torch.nn.Sigmoid()

    ite = iter(data_loader)
    for _ in range(6):
        color_img_pil, depth_img_pil, push_target_img_pil, grasp_target_img_pil = next(ite)
    color_img_pil_train = color_img_pil.to(device)
    depth_img_pil_train = depth_img_pil.to(device)

    outputs = model(color_img_pil_train, depth_img_pil_train)
    # push = sig(outputs[0][0]).cpu()
    # grasp = sig(outputs[1][0]).cpu()
    push = outputs[0][0].cpu()
    grasp = outputs[1][0].cpu()
    push *= (1 / PUSH_Q)
    push[push > 1] = 1
    push[push < 0] = 0
    grasp *= (1 / GRASP_Q)
    grasp[grasp > 1] = 1
    grasp[grasp < 0] = 0

    new_push = push.clone()
    new_grasp = grasp.clone()
    new_push[new_push > 0.5] = 1
    new_push[new_push <= 0.5] = 0
    new_grasp[new_grasp > 0.5] = 1
    new_grasp[new_grasp <= 0.5] = 0

    to_pil = torchvision.transforms.ToPILImage()
    img1 = to_pil(color_img_pil[0])
    img2 = to_pil(depth_img_pil[0])
    img3 = to_pil(push_target_img_pil[0])
    img4 = to_pil(grasp_target_img_pil[0])
    img5 = to_pil(push)
    img6 = to_pil(grasp)
    img7 = to_pil(new_push)
    img8 = to_pil(new_grasp)

    titles = [
        'Color',
        'Depth',
        'Target_push',
        'Target_grasp',
        'predicted push',
        'predicted grasp',
        'binary predicted push',
        'binary predicted grasp']
    images = [img1, img2, img3, img4, img5, img6, img7, img8]

    for i in range(len(images)):
        plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    # plt.savefig('test_pre.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train foreground')

    parser.add_argument('--dataset_root', dest='dataset_root', action='store', help='Enter the path to the dataset')
    parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=30, help='Enter the epoch for training')
    parser.add_argument('--batch_size', dest='batch_size', action='store', type=int,
                        default=16, help='Enter the batchsize for training and testing')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='Testing and visualizing')
    parser.add_argument('--lr', dest='lr', action='store', type=float, default=1e-6, help='Enter the learning rate')
    parser.add_argument('--real_fine_tuning', dest='real_fine_tuning', action='store_true', default=False, help='')
    parser.add_argument('--fine_tuning_num', dest='fine_tuning_num', action='store', type=int, default=16500, help='1500 action, one action contains 11 images')
    parser.add_argument(
        '--resume',
        dest='resume',
        action='store_true',
        default=False,
        help='Enter the path to the dataset')

    args = parser.parse_args()

    if args.resume:
        args.epochs = 10
    else:
        args.fine_tuning_num = None

    if args.test:
        test()
    else:
        main(args)
