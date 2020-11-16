#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
from camera import Camera
import cv2
from subprocess import Popen, PIPE
import numpy as np
import utils

camera = Camera()
time.sleep(1)  # Give camera some time to load data

# def get_camera_to_robot_transformation(camera):
#     color_img, depth_img = camera.get_data()
#     cv2.imwrite("real/temp.jpg", color_img)
#     p = Popen(['./real/detect-from-file', "real/temp.jpg"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     output, err = p.communicate()
#     tag_info = output.decode("utf-8")
#     tag_info = tag_info.split("\n")[:4]
#     for i, info in enumerate(tag_info):
#         tag_info[i] = info.split(" ")
#     tag_info = np.array(tag_info, dtype=np.float32)
#     assert(tag_info.shape == (4, 3))
#     tag_loc_camera = tag_info
#     tag_loc_robot = {
#         22: (270.15, -637.0),
#         7: (255.35, -247.6),
#         4: (-272.7, -660.9),
#         2: (-289.8, -274.2)
#     }
#     camera_to_robot = cv2.getPerspectiveTransform(
#         np.float32([tag[1:] for tag in tag_loc_camera]),
#         np.float32([tag_loc_robot[tag[0]] for tag in tag_loc_camera]))
#     return camera_to_robot


while True:
    color_img, depth_img = camera.get_data()
    # cv2.imwrite("temp.jpg", color_img)
    # p = Popen(['./detect-from-file', "temp.jpg"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # output, err = p.communicate()
    # rc = p.returncode
    # tag_info = output.decode("utf-8")
    # print(tag_info)
    # tag_info = tag_info.split("\n")[:4]
    # print(tag_info)
    # for i, info in enumerate(tag_info):
    #     tag_info[i] = info.split(" ")
    # print(tag_info)
    # tag_info = np.array(tag_info, dtype=np.float32)
    # print(tag_info)
    # assert(tag_info.shape == (4, 3))
    # tag_loc_camera = tag_info
    # tag_loc_robot = {
    #     22: (270.15 / 1000, -637.0 / 1000),
    #     7: (255.35 / 1000, -247.6 / 1000),
    #     4: (-272.7 / 1000, -660.9 / 1000),
    #     2: (-289.8 / 1000, -274.2 / 1000)
    # }
    # print(np.array([tag[1:] for tag in tag_loc_camera]))
    # print(np.array([tag_loc_robot[tag[0]] for tag in tag_loc_camera]))
    # camera_to_robot = cv2.getPerspectiveTransform(
    #     np.float32([tag[1:] for tag in tag_loc_camera]),
    #     np.float32([tag_loc_robot[tag[0]] for tag in tag_loc_camera]))
    # print(camera_to_robot)

    plt.subplot(211)
    plt.imshow(color_img)
    plt.subplot(212)
    plt.imshow(depth_img)
    plt.show()
