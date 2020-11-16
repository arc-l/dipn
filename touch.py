#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from real.camera import Camera
from robot import Robot
from subprocess import Popen, PIPE


def get_camera_to_robot_transformation(camera):
    color_img, depth_img = camera.get_data()
    cv2.imwrite("real/temp.jpg", color_img)
    p = Popen(['./real/detect-from-file', "real/temp.jpg"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    tag_info = output.decode("utf-8")
    tag_info = tag_info.split("\n")[:4]
    for i, info in enumerate(tag_info):
        tag_info[i] = info.split(" ")
    print(tag_info)
    tag_info = np.array(tag_info, dtype=np.float32)
    assert(tag_info.shape == (4, 3))
    tag_loc_camera = tag_info
    tag_loc_robot = {
        22: (270.15 / 1000, -637.0 / 1000),
        7: (255.35 / 1000, -247.6 / 1000),
        4: (-272.7 / 1000, -660.9 / 1000),
        2: (-289.8 / 1000, -274.2 / 1000)
    }
    camera_to_robot = cv2.getPerspectiveTransform(
        np.float32([tag[1:] for tag in tag_loc_camera]),
        np.float32([tag_loc_robot[tag[0]] for tag in tag_loc_camera]))
    return camera_to_robot


# User options (change me)
# --------------- Setup options ---------------
tcp_host_ip = '100.127.7.223'  # IP and port to robot arm as TCP client (UR5)
tcp_host_ip = "172.19.97.157"
tcp_port = 30002
rtc_host_ip = '100.127.7.223'  # IP and port to robot arm as real-time client (UR5)
rtc_host_ip = "172.19.97.157"
rtc_port = 30003
# Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])
workspace_limits = np.asarray([[-0.237, 0.211], [-0.683, -0.235], [0.18, 0.4]])
# workspace_limits = np.asarray([[-0.224, 0.224], [-0.674, -0.226], [0.18, 0.4]])
# Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
tool_orientation = [2.22, -2.22, 0]
tool_orientation = [0, -3.14, 0]
# ---------------------------------------------


# Move robot to home pose
robot = Robot(False, None, None, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              False, None, None)
robot.open_gripper()

transformation_matrix = get_camera_to_robot_transformation(robot.camera)

# Slow down robot
robot.joint_acc = 1.4
robot.joint_vel = 1.05

# Callback function for clicking on OpenCV window
click_point_pix = ()
camera_color_img, camera_depth_img = robot.get_camera_data()


def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global camera, robot, click_point_pix
        click_point_pix = (x, y)

        # Get click point in camera coordinates
        # click_z = camera_depth_img[y][x] * robot.cam_depth_scale
        # click_x = np.multiply(x-robot.cam_intrinsics[0][2],click_z/robot.cam_intrinsics[0][0])
        # click_y = np.multiply(y-robot.cam_intrinsics[1][2],click_z/robot.cam_intrinsics[1][1])
        # if click_z == 0:
        #     return
        # click_point = np.asarray([click_x,click_y,click_z])
        # click_point.shape = (3,1)

        # # Convert camera to robot coordinates
        # # camera2robot = np.linalg.inv(robot.cam_pose)
        # camera2robot = robot.cam_pose
        # target_position = np.dot(camera2robot[0:3,0:3],click_point) + camera2robot[0:3,3:]

        # target_position = target_position[0:3,0]
        # print(target_position)

        camera_pt = np.array([x, y, 1])
        robot_pt = np.dot(transformation_matrix, camera_pt)
        robot_pt = np.array([robot_pt[0], robot_pt[1]]) / robot_pt[2]

        print([robot_pt[0], robot_pt[1], -0.1])
        print(robot.parse_tcp_state_data(robot.get_state(), "cartesian_info"))

        robot.move_to([robot_pt[0], robot_pt[1], 0.3], tool_orientation)


# Show color and depth frames
cv2.namedWindow('color')
cv2.setMouseCallback('color', mouseclick_callback)
cv2.namedWindow('depth')

while True:
    camera_color_img, camera_depth_img = robot.get_camera_data()
    bgr_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    if len(click_point_pix) != 0:
        bgr_data = cv2.circle(bgr_data, click_point_pix, 7, (0, 0, 255), 2)
    cv2.imshow('color', bgr_data)
    camera_depth_img[camera_depth_img < 0.19] = 0
    cv2.imshow('depth', camera_depth_img)

    if cv2.waitKey(1) == ord('c'):
        break

cv2.destroyAllWindows()
