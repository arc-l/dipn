from robot import Robot
from logger import Logger
import numpy as np
import utils
import cv2

logger = Logger(False, '')

workspace_limits = np.asarray([[-0.248, 0.238], [-0.674, -0.233], [0.18, 0.4]])
workspace_half_size = 0.210
# workspace_limits = np.asarray([[-0.005 -0.224,-0.005 +0.224], [-0.450 -0.224, -0.450 + 0.224], [0.18, 0.4]])
workspace_limits = np.asarray([[-0.224, 0.224], [-0.674, -0.226], [0.185, 0.4]])
workspace_limits = np.asarray([[-0.237, 0.211], [-0.683, -0.235], [0.178, 0.4]])

robot = Robot(False,
              'objects/blocks',
              10,
              workspace_limits,
              "172.19.97.157", 30002, "172.19.97.157", 30003,
              False, '', '')
i = 20
prev_valid_depth_heightmap = None
while True:

    # Get latest RGB-D image
    color_img, depth_img = robot.get_camera_data()

    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    color_heightmap, depth_heightmap = utils.get_heightmap(
        color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, 0.002)
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    # color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
    # # cv2.imshow('color', color_heightmap)

    # # cv2.imshow('depth', valid_depth_heightmap)

    print(np.max(valid_depth_heightmap))
    valid_depth_heightmap[valid_depth_heightmap < 0.01] = 0
    valid_depth_heightmap[valid_depth_heightmap >= 0.01] = 255
    print(np.sum(valid_depth_heightmap == 255))

    # cv2.imshow('binary depth', valid_depth_heightmap)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save RGB-D images and RGB-D heightmaps
    # logger.save_images(i, color_img, depth_img, '0')
    # logger.save_heightmaps(i, color_heightmap, valid_depth_heightmap, '0')
    i += 1

    # if prev_valid_depth_heightmap is not None:
    #     depth_diff = abs(valid_depth_heightmap - prev_valid_depth_heightmap)
    #     depth_diff[np.isnan(depth_diff)] = 0
    #     depth_diff[depth_diff > 0.3] = 0
    #     depth_diff[depth_diff < 0.01] = 0
    #     depth_diff[depth_diff > 0] = 1
    #     print('change', np.sum(depth_diff))

    # prev_valid_depth_heightmap = valid_depth_heightmap

    input('refresh image')
