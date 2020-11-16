#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time
from vision.backbone_utils import resnet_fpn_net
from constants import is_real

class reactive_net(nn.Module):
    """
    Haven't completed
    """
    def __init__(self, use_cuda, pre_train=False):  # , snapshot=None
        super(reactive_net, self).__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda")
        self.pre_train = pre_train

        self.num_rotations = 16

        if pre_train:
            self.pushnet = resnet_fpn_net('resnet18', grasp=False).to(self.device)
            self.graspnet = resnet_fpn_net('resnet18').to(self.device)
        else:
            self.pushnet = resnet_fpn_net('resnet18', trainable_layers=5, grasp=False).to(self.device)
            self.graspnet = resnet_fpn_net('resnet18', trainable_layers=5).to(self.device)

        # self.sig = torch.nn.Sigmoid().to(self.device)

        print("max_memory_allocated (MB):", torch.cuda.max_memory_allocated() / 2**20)
        print("memory_allocated (MB):", torch.cuda.memory_allocated() / 2**20)

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1, push_pred=False):

        if self.pre_train:
            input_data = torch.cat((input_color_data, input_depth_data), dim=1)

            final_push_feat = self.pushnet(input_data)
            final_grasp_feat = self.graspnet(input_data)

            output_probs = (final_push_feat, final_grasp_feat)

            return output_probs

        else:
            if is_volatile:
                with torch.no_grad():
                    output_prob = []

                    # Apply rotations to images
                    for rotate_idx in range(self.num_rotations):
                        rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                        # Compute sample grid for rotation BEFORE neural network
                        affine_mat_before = np.asarray(
                            [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                        affine_mat_before.shape = (2, 3, 1)
                        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
                        flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=True)

                        # Rotate images clockwise
                        rotate_color = F.grid_sample(
                            input_color_data.to(
                                self.device),
                            flow_grid_before,
                            mode='nearest',
                            align_corners=True)
                        rotate_depth = F.grid_sample(
                            input_depth_data.to(
                                self.device),
                            flow_grid_before,
                            mode='nearest',
                            align_corners=True)

                        input_data = torch.cat((rotate_color, rotate_depth), dim=1)

                        # Pass intermediate features to net
                        if push_pred:
                            final_grasp_feat = self.graspnet(input_data)
                        else:
                            final_push_feat = self.pushnet(input_data)
                            final_grasp_feat = self.graspnet(input_data)

                        # Compute sample grid for rotation AFTER branches
                        affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(
                            rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                        affine_mat_after.shape = (2, 3, 1)
                        affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
                        flow_grid_after = F.affine_grid(
                            affine_mat_after, final_grasp_feat.data.size(), align_corners=True)

                        # Forward pass through branches, undo rotation on output predictions, upsample results
                        if push_pred:
                            output_prob.append([None,
                                                F.grid_sample(final_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)])
                        else:
                            output_prob.append([F.grid_sample(final_push_feat, flow_grid_after, mode='nearest', align_corners=True),
                                                F.grid_sample(final_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)])

                return output_prob

            else:
                self.output_prob = []

                # Apply rotations to images
                rotate_idx = specific_rotation
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray(
                    [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
                affine_mat_before.requires_grad_(False)
                flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=True)

                # Rotate images clockwise
                rotate_color = F.grid_sample(
                    input_color_data.to(
                        self.device),
                    flow_grid_before,
                    mode='nearest',
                    align_corners=True)
                rotate_depth = F.grid_sample(
                    input_depth_data.to(
                        self.device),
                    flow_grid_before,
                    mode='nearest',
                    align_corners=True)

                input_data = torch.cat((rotate_color, rotate_depth), dim=1)

                # Pass intermediate features to net
                final_push_feat = self.pushnet(input_data)
                final_grasp_feat = self.graspnet(input_data)

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                               [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
                affine_mat_after.requires_grad_(False)
                flow_grid_after = F.affine_grid(
                    affine_mat_after.to(
                        self.device),
                    final_push_feat.data.size(),
                    align_corners=True)

                # Forward pass through branches, undo rotation on output predictions, upsample results
                self.output_prob.append([F.grid_sample(final_push_feat, flow_grid_after, mode='nearest', align_corners=True),
                                         F.grid_sample(final_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)])

                return self.output_prob



class reinforcement_net(nn.Module):
    """
    The DQN Network.
    graspnet is the Grasp Network.
    pushnet is the Push Network for the DQN + GN method.
    """
    def __init__(self, use_cuda, pre_train=False):  # , snapshot=None
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda")
        self.pre_train = pre_train

        self.num_rotations = 16

        if pre_train:
            self.pushnet = resnet_fpn_net('resnet18', grasp=False, is_real=is_real).to(self.device)
            self.graspnet = resnet_fpn_net('resnet18', is_real=is_real).to(self.device)
        else:
            self.pushnet = resnet_fpn_net('resnet18', trainable_layers=5, grasp=False, is_real=is_real).to(self.device)
            self.graspnet = resnet_fpn_net('resnet18', trainable_layers=5, is_real=is_real).to(self.device)

        print("max_memory_allocated (MB):", torch.cuda.max_memory_allocated() / 2**20)
        print("memory_allocated (MB):", torch.cuda.memory_allocated() / 2**20)

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1, use_push=True):

        if self.pre_train:
            input_data = torch.cat((input_color_data, input_depth_data), dim=1)

            final_push_feat = self.pushnet(input_data)
            final_grasp_feat = self.graspnet(input_data)

            output_probs = (final_push_feat, final_grasp_feat)

            return output_probs

        else:
            if is_volatile:
                with torch.no_grad():
                    output_prob = []

                    # Apply rotations to images
                    for rotate_idx in range(self.num_rotations):
                        rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                        # Compute sample grid for rotation BEFORE neural network
                        affine_mat_before = np.asarray(
                            [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                        affine_mat_before.shape = (2, 3, 1)
                        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
                        flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=True)

                        # Rotate images clockwise
                        rotate_color = F.grid_sample(
                            input_color_data.to(
                                self.device),
                            flow_grid_before,
                            mode='nearest',
                            align_corners=True)
                        rotate_depth = F.grid_sample(
                            input_depth_data.to(
                                self.device),
                            flow_grid_before,
                            mode='nearest',
                            align_corners=True)

                        input_data = torch.cat((rotate_color, rotate_depth), dim=1)

                        # Pass intermediate features to net
                        if use_push:
                            final_push_feat = self.pushnet(input_data)
                            final_grasp_feat = self.graspnet(input_data)
                        else:
                            final_grasp_feat = self.graspnet(input_data)


                        # Compute sample grid for rotation AFTER branches
                        affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(
                            rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                        affine_mat_after.shape = (2, 3, 1)
                        affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
                        flow_grid_after = F.affine_grid(
                            affine_mat_after, final_grasp_feat.data.size(), align_corners=True)

                        # Forward pass through branches, undo rotation on output predictions, upsample results
                        if use_push:
                            output_prob.append([F.grid_sample(final_push_feat, flow_grid_after, mode='nearest', align_corners=True),
                                                F.grid_sample(final_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)])
                        else:
                            output_prob.append([None,
                                                F.grid_sample(final_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)])

                return output_prob

            else:
                self.output_prob = []

                # Apply rotations to images
                rotate_idx = specific_rotation
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                # Compute sample grid for rotation BEFORE branches
                affine_mat_before = np.asarray(
                    [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
                affine_mat_before.requires_grad_(False)
                flow_grid_before = F.affine_grid(affine_mat_before, input_color_data.size(), align_corners=True)

                # Rotate images clockwise
                rotate_color = F.grid_sample(
                    input_color_data.to(
                        self.device),
                    flow_grid_before,
                    mode='nearest',
                    align_corners=True)
                rotate_depth = F.grid_sample(
                    input_depth_data.to(
                        self.device),
                    flow_grid_before,
                    mode='nearest',
                    align_corners=True)

                input_data = torch.cat((rotate_color, rotate_depth), dim=1)

                # Pass intermediate features to net
                final_push_feat = self.pushnet(input_data)
                final_grasp_feat = self.graspnet(input_data)

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                               [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
                affine_mat_after.requires_grad_(False)
                flow_grid_after = F.affine_grid(
                    affine_mat_after.to(
                        self.device),
                    final_push_feat.data.size(),
                    align_corners=True)

                # Forward pass through branches, undo rotation on output predictions, upsample results
                self.output_prob.append([F.grid_sample(final_push_feat, flow_grid_after, mode='nearest', align_corners=True),
                                         F.grid_sample(final_grasp_feat, flow_grid_after, mode='nearest', align_corners=True)])

                return self.output_prob
