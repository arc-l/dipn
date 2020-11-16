import torch
import torch.nn as nn
import torch.nn.functional as F
from vision.resnet import ResNetSmallNoFC, Bottleneck, BasicBlock
from vision.backbone_utils import resent_backbone
from collections import OrderedDict
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from collections import OrderedDict
from torchvision.ops import misc as misc_nn_ops
from constants import total_obj




class PushPredictionNet(nn.Module):

    def __init__(self):
        super().__init__()

        # single object state encoder
        self.single_state_encoder = nn.Sequential(OrderedDict([
            ('single-state-encoder-fc1', nn.Linear(2, 8)),
            ('single-state-encoder-relu1', nn.ReLU(inplace=True)),
            ('single-state-encoder-fc2', nn.Linear(8, 16)),
            ('single-state-encoder-relu2', nn.ReLU(inplace=True))
        ]))

        # single object image encoder
        self.singel_image_encoder = resent_backbone('resnet18', pretrained=False, num_classes=64, input_channels=1)

        # Interactive transformation
        self.interact = nn.Sequential(OrderedDict([
            ('interact-fc1', nn.Linear(176, 128)),
            ('interact-relu1', nn.ReLU(inplace=True)),
            ('interact-fc2', nn.Linear(128, 128)),
            ('interact-relu2', nn.ReLU(inplace=True)),
            ('interact-fc3', nn.Linear(128, 128)),
            ('interact-relu3', nn.ReLU(inplace=True))
        ]))

        # Direct transformation
        self.dynamics = nn.Sequential(OrderedDict([
            ('dynamics-fc1', nn.Linear(96, 128)),
            ('dynamics-relu1', nn.ReLU(inplace=True)),
            ('dynamics-fc2', nn.Linear(128, 128)),
            ('dynamics-relu2', nn.ReLU(inplace=True))
        ]))

        # action encoder
        self.action_encoder = nn.Sequential(OrderedDict([
            ('action_encoder-fc1', nn.Linear(4, 8)),
            ('action_encoder-relu1', nn.ReLU(inplace=True)),
            ('action_encoder-fc2', nn.Linear(8, 16)),
            ('action_encoder-relu2', nn.ReLU(inplace=True))
        ]))

        # global image encoder
        self.image_encoder = resent_backbone('resnet18', pretrained=False, num_classes=512, input_channels=2)

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder-fc00', nn.Linear(640, 256)),
            ('decoder-relu00', nn.ReLU(inplace=True)),
            ('decoder-fc0', nn.Linear(256, 64)),
            ('decoder-relu0', nn.ReLU(inplace=True)),
            ('decoder-fc1', nn.Linear(64, 32)),
            ('decoder-relu1', nn.ReLU(inplace=True)),
            ('decoder-fc2', nn.Linear(32, 16)),
            ('decoder-relu2', nn.ReLU(inplace=True)),
            ('decoder-fc3', nn.Linear(16, 3))
        ]))


    def forward(self, prev_poses, action, image, image_objs, num_objs):

        # action
        encoded_action = self.action_encoder(action)

        # single object
        encoded_info = []
        for i in range(num_objs):
            encoded_state = self.single_state_encoder(prev_poses[:, i*2:i*2+2])
            encoded_image = self.singel_image_encoder(image_objs[:, i:i+1, :, :,])
            encoded_cat = torch.cat((encoded_state, encoded_image), dim=1)
            encoded_info.append(encoded_cat)

        # the environment
        y = self.image_encoder(image)

        # interact
        z = None
        for i in range(num_objs):
            dy_input = torch.cat((encoded_action, encoded_info[i]), dim=1)
            all_dynamics = self.dynamics(dy_input)
            for j in range(1, num_objs):
                idx = i + j
                if idx >= num_objs: idx = idx - num_objs
                inter_input = torch.cat((dy_input, encoded_info[idx]), dim=1)
                other = self.interact(inter_input)
                all_dynamics = all_dynamics + other
            de_input = torch.cat((y, all_dynamics), dim=1)
            output = self.decoder(de_input)
            if z is None:
                z = output
            else:
                z = torch.cat((z, output), dim=1)

        return z