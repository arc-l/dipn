import socket
import select
import struct
import time
import os
import numpy as np
import utils
from simulation import sim as vrep
from sys import exit
import math
import random
import atexit
import cv2
import collections
import traceback


background_threshold = {"low": np.array([0, 0, 120], np.uint8), "high": np.array([255, 255, 255], np.uint8)}


class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file, collect_push=False):

        self.is_sim = is_sim
        self.workspace_limits = workspace_limits

        # If in simulation...
        if self.is_sim:

            # Define colors for object meshes (Tableau palette)
            self.color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                                           [89.0, 161.0, 79.0],  # green
                                           [156, 117, 95],  # brown
                                           [242, 142, 43],  # orange
                                           [237.0, 201.0, 72.0],  # yellow
                                           [186, 176, 172],  # gray
                                           [255.0, 87.0, 89.0],  # red
                                           [176, 122, 161],  # purple
                                           [118, 183, 178],  # cyan
                                           [255, 157, 167]]) / 255.0  # pink

            # Read files in object mesh directory
            self.obj_mesh_dir = obj_mesh_dir
            self.num_obj = num_obj
            self.mesh_list = os.listdir(self.obj_mesh_dir)
            # TODO: disable this while training VPG
            # 0 - traingle, 1 - non-convex, 2 - square, 3 - thin rectangle, 4 - cube, 6 - rectangle, 7 - half cylinder, 8 - cylinder
            # self.mesh_list = ['0.obj', '1.obj', '2.obj', '3.obj', '4.obj', '6.obj', '7.obj', '8.obj']

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]
            # TODO: only used for create.py
            # self.obj_mesh_ind = np.array([1, 1, 0, 0, 4, 5])

            # Make sure to have the server side running in V-REP:
            # in a child script of a V-REP scene, add following command
            # to be executed just once, at simulation start:
            #
            # simExtRemoteApiStart(19999)
            #
            # then start simulation, and run this program.
            #
            # IMPORTANT: for each successful call to simxStart, there
            # should be a corresponding call to simxFinish at the end!

            # MODIFY remoteApiConnections.txt

            # Connect to simulator
            vrep.simxFinish(-1)  # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            # If testing, read object meshes and poses from test case file
            if self.is_testing and self.test_preset_cases:
                file = open(self.test_preset_file, 'r')
                file_content = file.readlines()
                self.test_obj_mesh_files = []
                self.test_obj_mesh_colors = []
                self.test_obj_positions = []
                self.test_obj_orientations = []
                for object_idx in range(self.num_obj):
                    file_content_curr_object = file_content[object_idx].split()
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir, file_content_curr_object[0]))
                    self.test_obj_mesh_colors.append([float(file_content_curr_object[1]), float(
                        file_content_curr_object[2]), float(file_content_curr_object[3])])
                    self.test_obj_positions.append([float(file_content_curr_object[4]), float(
                        file_content_curr_object[5]), float(file_content_curr_object[6])])
                    self.test_obj_orientations.append([float(file_content_curr_object[7]), float(
                        file_content_curr_object[8]), float(file_content_curr_object[9])])
                file.close()
                self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

            # Add objects to simulation environment
            if not collect_push:
                self.add_objects()

        # If in real-settings...
        else:
            # Connect to robot client
            self.tcp_host_ip = tcp_host_ip
            self.tcp_port = tcp_port
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            self.tcp_socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket_gripper.connect((self.tcp_host_ip, 63352))
            self.rtc_host_ip = rtc_host_ip
            self.rtc_port = rtc_port
            self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))
            atexit.register(self.tcp_socket.close)
            atexit.register(self.tcp_socket_gripper.close)
            atexit.register(self.rtc_socket.close)

            # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Connect as real-time client to parse state data

            # Default home joint configuration
            self.home_joint_config = [12.58, -127.32, 127.42, -90.1, -89.83, -81.49]
            self.home_joint_config = [j / 180 * np.pi for j in self.home_joint_config]

            # Default joint speed configuration
            self.joint_acc = 8  # Safe: 1.4
            self.joint_vel = 3  # Safe: 1.05
            # self.joint_acc = 0.5  # Safe: 1.4
            # self.joint_vel = 0.5  # Safe: 1.05

            # Joint tolerance for blocking calls
            self.joint_tolerance = 0.01

            # Default tool speed configuration
            self.tool_acc = 1.2  # Safe: 0.5
            self.tool_vel = 0.3  # Safe: 0.2

            # Tool pose tolerance for blocking calls
            self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

            # Move robot to home pose
            self.go_home()
            self.setup_gripper()
            self.close_gripper()

            # Fetch RGB-D data from RealSense camera
            from real.camera import Camera
            self.camera = Camera()
            self.cam_intrinsics = self.camera.intrinsics

            # Load camera pose (from running calibrate.py), intrinsics and depth scale
            self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

    def setup_sim_camera(self):
        """
        cam_intrinsics (in degree)

        ratio=resolutionX/resolutionY
        if (ratio>1)
        {
            angleX=perspectiveAngle
            angleY=2*atan(tan(perspectiveAngle/2)/ratio)
        }
        else
        {
            angleX=2*atan(tan(perspectiveAngle/2)*ratio)
            angleY=perspectiveAngle
        }

        focal length = w/2 * cot(angle/2)
        """

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_ortho', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(
            self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(
            self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
        # self.cam_intrinsics = np.asarray([[618.66, 0, 320], [0, 614.57, 240], [0, 0, 1]]) # 640*480
        # self.cam_intrinsics = np.asarray([[773.33, 0, 400], [0, 773.47, 300], [0, 0, 1]]) # 800*600
        self.cam_intrinsics = np.asarray([[443.405, 0, 256], [0, 443.405, 256], [0, 0, 1]])  # 512*512
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def add_objects_mask(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * \
                np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * \
                np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [
                    self.test_obj_positions[object_idx][0],
                    self.test_obj_positions[object_idx][1],
                    self.test_obj_positions[object_idx][2]]
                object_orientation = [
                    self.test_obj_orientations[object_idx][0],
                    self.test_obj_orientations[object_idx][1],
                    self.test_obj_orientations[object_idx][2]]
            if object_idx % 2 == 0:
                color_index = random.randint(0, len(self.obj_mesh_color) - 1)
                object_color = [
                    self.obj_mesh_color[color_index][0] + random.randint(-20, 20) / 255,
                    self.obj_mesh_color[color_index][1] + random.randint(-20, 20) / 255,
                    self.obj_mesh_color[color_index][2] + random.randint(-20, 20) / 255]
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                self.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'importShape', [
                    0, 0, 255, 0], object_position + object_orientation + object_color, [
                    curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(0.1)
        self.prev_obj_positions = []
        self.obj_positions = []

        num_list = [1]
        for i in range(2, self.num_obj):
            num_list.extend([i] * i)
        self.num_obj = random.choice(num_list)

    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * \
                np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * \
                np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            # TODO: switch this while training VPG
            # object_orientation = [0, 0, 2*np.pi*np.random.random_sample()]
            object_orientation = [
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [
                    self.test_obj_positions[object_idx][0],
                    self.test_obj_positions[object_idx][1],
                    self.test_obj_positions[object_idx][2]]
                object_orientation = [
                    self.test_obj_orientations[object_idx][0],
                    self.test_obj_orientations[object_idx][1],
                    self.test_obj_orientations[object_idx][2]]
            object_color = [
                self.obj_mesh_color[object_idx][0],
                self.obj_mesh_color[object_idx][1],
                self.obj_mesh_color[object_idx][2]]
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                self.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'importShape', [
                    0, 0, 255, 0], object_position + object_orientation + object_color, [
                    curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            if not (self.is_testing and self.test_preset_cases):
                time.sleep(1)
        time.sleep(1)
        self.prev_obj_positions = []
        self.obj_positions = []

    def add_object_push(self):
        drop_height = 0.1
        obj_mesh_ind = self.mesh_list
        obj_num = np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.02, 0.05, 0.2, 0.35, 0.3, 0.05, 0.03])
        obj_mesh_ind = np.random.choice(obj_mesh_ind, obj_num)
        obj_mesh_color_ind = [0, 1, 2, 3, 4, 5, 6]

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        object_orientations = []
        bbox_heights = []
        for object_idx in range(len(obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, obj_mesh_ind[object_idx])
            curr_shape_name = 'shape_%02d' % object_idx
            # drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.3) * np.random.random_sample() + self.workspace_limits[0][0] + 0.2
            # drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.3) * np.random.random_sample() + self.workspace_limits[1][0] + 0.2
            drop_x = -0.6 + np.random.random_sample() * 0.1
            drop_y = -0.05 + np.random.random_sample() * 0.1
            object_position = [drop_x, drop_y, drop_height]
            # object_orientation = [np.pi / 2, 2*np.pi*np.random.random_sample(), np.pi / 2]
            object_orientation = [0, 0, 2 * np.pi * np.random.random_sample()]
            object_orientations.append(object_orientation)
            object_color = [self.obj_mesh_color[obj_mesh_color_ind[object_idx]][0],
                            self.obj_mesh_color[obj_mesh_color_ind[object_idx]][1],
                            self.obj_mesh_color[obj_mesh_color_ind[object_idx]][2]]
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                self.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'importShape', [
                    0, 0, 255, 0], object_position + object_orientation + object_color, [
                    curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]

            return_code, height = vrep.simxGetObjectFloatParameter(
                self.sim_client, curr_shape_handle, vrep.sim_objfloatparam_objbbox_max_z, vrep.simx_opmode_blocking)
            # if height < 0.02:  # 0.0135 and 0.0165
            #     height = 0.017
            # else:  # 0.025
            #     height = 0.03
            # height += 0.01
            height = height / 2 + 0.023
            bbox_heights.append(height)

            count = 0
            while True:
                time.sleep(0.3)
                if count > 10:
                    break
                sim_ret, object_position = vrep.simxGetObjectPosition(
                    self.sim_client, curr_shape_handle, -1, vrep.simx_opmode_blocking)
                # if overlap
                if object_position[2] > height:
                    drop_x = math.cos(object_orientation[2]) * 0.025 + object_position[0]
                    drop_y = math.sin(object_orientation[2]) * 0.025 + object_position[1]
                    object_position = [drop_x, drop_y, drop_height]
                    vrep.simxSetObjectPosition(self.sim_client, curr_shape_handle, -1,
                                               object_position, vrep.simx_opmode_blocking)
                    vrep.simxSetObjectOrientation(self.sim_client, curr_shape_handle, -1,
                                                  object_orientation, vrep.simx_opmode_blocking)
                else:
                    break
                count += 1
            if count > 10:
                object_position = [drop_x, drop_y, height]
                vrep.simxSetObjectPosition(self.sim_client, curr_shape_handle, -1,
                                           object_position, vrep.simx_opmode_blocking)

            self.object_handles.append(curr_shape_handle)
            time.sleep(0.1)

        for idx in range(len(self.object_handles)):
            vrep.simxSetObjectOrientation(
                self.sim_client, self.object_handles[idx], -1, object_orientations[idx], vrep.simx_opmode_blocking)
            time.sleep(0.1)
        time.sleep(0.5)

        self.prev_obj_positions = []
        self.obj_positions = []

        return bbox_heights

    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(
            self.sim_client, 'UR5_target', vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                   1, (-0.5, 0, 0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(
            self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(
                self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)

        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)

    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(
            self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - \
            0.1 and gripper_position[1] < self.workspace_limits[1][1] + \
            0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()

    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0],  # red
                                    [-0.625, -0.125, 0.0],  # blue
                                    [-0.375, 0.125, 0.0],  # green
                                    [-0.375, -0.125, 0.0]])  # yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1, obj_positions.shape[1], 1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)

    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached

    # def stop_sim(self):
    #     if self.is_sim:
    #         # Now send some data to V-REP in a non-blocking fashion:
    #         # vrep.simxAddStatusbarMessage(sim_client,'Hello V-REP!',vrep.simx_opmode_oneshot)

    #         # # Start the simulation
    #         # vrep.simxStartSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

    #         # # Stop simulation:
    #         # vrep.simxStopSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

    #         # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    #         vrep.simxGetPingTime(self.sim_client)

    #         # Now close the connection to V-REP:
    #         vrep.simxFinish(self.sim_client)

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(
                self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(
                self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(
                self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * \
                np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * \
                np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1,
                                          object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)

    def get_camera_data(self):

        if self.is_sim:

            # Get color image from simulation
            sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(
                self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
            color_img = np.asarray(raw_image)
            color_img.shape = (resolution[1], resolution[0], 3)
            color_img = color_img.astype(np.float) / 255
            color_img[color_img < 0] += 1
            color_img *= 255
            color_img = np.fliplr(color_img)
            color_img = color_img.astype(np.uint8)

            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(
                self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

        else:
            # Get color and depth image from ROS service
            color_img, depth_img = self.camera.get_data()
            # Remove background
            img = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
            bg_mask = cv2.inRange(img, background_threshold["low"], background_threshold["high"])
            color_img = cv2.bitwise_and(color_img, color_img, mask=bg_mask)

        return color_img, depth_img

    def parse_tcp_state_data(self, state_data, subpackage):

        # Read package header
        robot_message_type = 20
        while robot_message_type != 16:
            state_data = self.get_state()
            data_bytes = bytearray()
            data_bytes.extend(state_data)
            data_length = struct.unpack("!i", data_bytes[0:4])[0]
            # print("package length", data_length)
            robot_message_type = data_bytes[4]
        assert(robot_message_type == 16)

        byte_idx = 5
        # Parse sub-packages
        subpackage_types = {'joint_data': 1, 'cartesian_info': 4, 'force_mode_data': 7, 'tool_data': 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx + 4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            # print(package_idx)
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0, 0, 0, 0, 0, 0]
            target_joint_positions = [0, 0, 0, 0, 0, 0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 8):(byte_idx + 16)])[0]
                byte_idx += 41
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                byte_idx += 8
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            return tool_analog_input2

        def parse_force_mode_data(data_bytes, byte_idx):
            forces = [0, 0, 0, 0, 0, 0]
            for force_idx in range(6):
                forces[force_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                byte_idx += 8
            return forces

        parse_functions = {
            'joint_data': parse_joint_data,
            'cartesian_info': parse_cartesian_info,
            'tool_data': parse_tool_data,
            'force_mode_data': parse_force_mode_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def parse_rtc_state_data(self, state_data):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        # data_length = struct.unpack("!i", data_bytes[0:4])[0]
        # print("RTC")
        # print(struct.unpack("!i", data_bytes[0:4])[0])
        # print(struct.unpack("!i", data_bytes[4:8])[0])
        # assert(data_length == 812)
        # byte_idx = 4 + 8 + 8 * 48 + 24 + 120

        tcp_pose = [0 for i in range(6)]
        tcp_force = [0 for i in range(6)]
        byte_idx = 444
        for joint_idx in range(6):
            tcp_pose[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            byte_idx += 8
        byte_idx = 540
        for joint_idx in range(6):
            tcp_force[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            byte_idx += 8

        return tcp_pose, tcp_force

    def close_gripper(self, asynch=False):

        if self.is_sim:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(
                self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(
                self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(
                self.sim_client,
                RG2_gripper_handle,
                gripper_motor_velocity,
                vrep.simx_opmode_blocking)
            gripper_fully_closed = False
            while gripper_joint_position > -0.045:  # Block until gripper is fully closed
                sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(
                    self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                # print(gripper_joint_position)
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True

        else:
            tcp_command = "SET POS 255\n"
            self.tcp_socket_gripper.close()
            self.tcp_socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket_gripper.connect((self.tcp_host_ip, 63352))
            self.tcp_socket_gripper.send(str.encode(tcp_command))
            self.tcp_socket_gripper.recv(8)

            max_gripper_pos = self.get_gripper_pos()
            counter = 0
            while True:
                time.sleep(0.005)
                next_gripper_pos = self.get_gripper_pos()
                if next_gripper_pos <= max_gripper_pos:
                    counter += 1
                else:
                    counter = 0
                if counter > 20:
                    break
                max_gripper_pos = max([next_gripper_pos, max_gripper_pos])

            return True
        return gripper_fully_closed

    def open_gripper(self, asynch=False):

        if self.is_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(
                self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(
                self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(
                self.sim_client,
                RG2_gripper_handle,
                gripper_motor_velocity,
                vrep.simx_opmode_blocking)
            while gripper_joint_position < 0.03:  # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(
                    self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)

        else:
            tcp_command = "SET POS 0\n"
            self.tcp_socket_gripper.close()
            self.tcp_socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket_gripper.connect((self.tcp_host_ip, 63352))
            self.setup_gripper()
            self.tcp_socket_gripper.send(str.encode(tcp_command))
            self.tcp_socket_gripper.recv(8)

            prev_gripper_pos = self.get_gripper_pos()
            while True:
                time.sleep(0.2)
                next_gripper_pos = self.get_gripper_pos()
                if next_gripper_pos >= prev_gripper_pos:
                    break
                prev_gripper_pos = next_gripper_pos

            # if not asynch:
            #     time.sleep(1.5)

    def get_gripper_pos(self):
        tcp_command = "GET POS\n"
        self.tcp_socket_gripper.send(str.encode(tcp_command))
        info = self.tcp_socket_gripper.recv(8).decode("utf-8").split()
        current_pose = int(info[-1])
        return current_pose

    def setup_gripper_low(self):
        self.tcp_socket_gripper.send(str.encode("SET FOR 100\n"))
        self.tcp_socket_gripper.recv(8)
        self.tcp_socket_gripper.send(str.encode("SET SPE 120\n"))
        self.tcp_socket_gripper.recv(8)

    def setup_gripper(self):
        self.tcp_socket_gripper.send(str.encode("SET FOR 255\n"))
        self.tcp_socket_gripper.recv(8)
        self.tcp_socket_gripper.send(str.encode("SET SPE 255\n"))
        self.tcp_socket_gripper.recv(8)

    def get_state(self):
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(2048)
        state_data = self.tcp_socket.recv(2048)
        # self.tcp_socket.close()
        return state_data

    def reset_plane(self):
        if self.is_sim:
            sim_ret, plane_handle = vrep.simxGetObjectHandle(
                self.sim_client, 'Box', vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, plane_handle, -1,
                                                [-10, -10, 0.5], vrep.simx_opmode_blocking)

    def move_to(self, tool_position, tool_orientation, speed_scale=1.0):

        if self.is_sim:

            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

            move_direction = np.asarray([tool_position[0] -
                                         UR5_target_position[0], tool_position[1] -
                                         UR5_target_position[1], tool_position[2] -
                                         UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                           1, (UR5_target_position[0] +
                                               move_step[0], UR5_target_position[1] +
                                               move_step[1], UR5_target_position[2] +
                                               move_step[2]), vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(
                    self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                       1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)

        else:
            # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],
                                                                               tool_position[1],
                                                                               tool_position[2],
                                                                               tool_orientation[0],
                                                                               tool_orientation[1],
                                                                               tool_orientation[2],
                                                                               self.tool_acc * speed_scale,
                                                                               self.tool_vel * speed_scale)
            self.tcp_socket.send(str.encode(tcp_command))

            # Block until robot reaches target tool position
            tcp_state_data = self.tcp_socket.recv(2048)
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            self.tcp_socket.settimeout(0.5)
            while not all([np.abs(actual_tool_pose[j] - tool_position[j]) <
                           self.tool_pose_tolerance[j] for j in range(3)]):
                # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
                # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)]
                # + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]),
                # np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2))
                # for j in range(3,6)])
                try:
                    tcp_state_data = self.tcp_socket.recv(2048)
                    prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
                    actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                # except socket.timeout:
                except:
                    print("TCP socket Timeout!!!!")
                    self.tcp_socket.close()
                    self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                    self.tcp_socket.settimeout(0.5)
                time.sleep(0.01)
            time.sleep(0.2)
            # self.tcp_socket.close()

    def protected_move_to(self, tool_position, tool_orientation, speed_scale=1.0, force_max=50):
        tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],
                                                                           tool_position[1],
                                                                           tool_position[2],
                                                                           tool_orientation[0],
                                                                           tool_orientation[1],
                                                                           tool_orientation[2],
                                                                           self.tool_acc * speed_scale,
                                                                           self.tool_vel * speed_scale)
        self.tcp_socket.send(str.encode(tcp_command))
        # self.rtc_socket.settimeout(1)
        self.rtc_socket.close()
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))
        pose, force = self.parse_rtc_state_data(self.rtc_socket.recv(1108))
        self.rtc_socket.settimeout(0.5)
        pose_history = collections.deque([pose], 100)
        start_time = time.time()
        max_force = 0
        print('Protected move to...')
        # try:
        while not all([np.abs(pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            try:
                pose, force = self.parse_rtc_state_data(self.rtc_socket.recv(1108))
            except socket.timeout:
                print("RTC socket Timeout!!!!")
                self.rtc_socket.close()
                self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))
                self.rtc_socket.settimeout(0.5)
            if np.linalg.norm(np.asarray(force[0: 3])) > force_max:
                print("Collision Found!!!!", np.linalg.norm(np.asarray(force[0: 3])))
                self.move_to(pose_history[0], [tool_orientation[0],
                                               tool_orientation[1], tool_orientation[2]], speed_scale)
                return False
            else:
                max_force = max(max_force, np.linalg.norm(np.asarray(force[0: 3])))
            pose_history.append(pose)
            # time.sleep(0.05)
            time.sleep(0.001)
            if time.time() - start_time > 6:
                print('TIMEOUT!!!!!!!!')
                return False
        # except Exception as ex:
        #     prnit('?????????????????????????????')
        #     traceback.print_exception(type(ex), ex, ex.__traceback__)
        # while not all([np.abs(pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
        #     pose, force = self.parse_rtc_state_data(self.rtc_socket.recv(1108))
        #     if np.linalg.norm(np.asarray(force[0: 3])) > force_max:
        #         print("Collision Found!!!!")
        #         self.move_to(pose_history[0], [tool_orientation[0],
        #                                        tool_orientation[1], tool_orientation[2]], speed_scale)
        #         return False
        #     pose_history.append(pose)
        #     # time.sleep(0.05)
        #     time.sleep(0.001)
        #     if time.time() - start_time > 5:
        #         print('TIMEOUT!!!!!!!!')
        #         return False
        time.sleep(0.2)
        print("Max Force during this move", max_force)

        return True

    def compliant_move_to(self, tool_position, tool_orientation, speed_scale=1.0):
        # force_command = "rq_activate_and_wait()\n"
        # self.tcp_socket.send(str.encode(force_command))
        # force_command = "def mvfce():\n"
        force_command = "thread Force_properties_calculation_thread_1()\n"
        force_command += "while (True):\n"
        force_command += "force_mode (tool_pose(), [0, 0, 1, 0, 0, 0], [0.0, 0.0, 10.0, 0.0, 0.0, 0.0], 2, [0.1, 0.1, 0.15, 0.17, 0.17, 0.17])\n"
        force_command += "sync()\n"
        force_command += "end\n"
        force_command += "end\n"
        force_command += "global thread_handler_1=run Force_properties_calculation_thread_1()\n"
        force_command += "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],
                                                                              tool_position[1],
                                                                              tool_position[2],
                                                                              tool_orientation[0],
                                                                              tool_orientation[1],
                                                                              tool_orientation[2],
                                                                              self.tool_acc * speed_scale,
                                                                              self.tool_vel * speed_scale)
        force_command += "kill thread_handler_1\n"
        # force_command += "end_force_mode()\n"
        # force_command += "end\n"
        # self.tcp_socket.send(str.encode(force_command))
        # force_command = "mvfce()\n"
        self.tcp_socket.send(str.encode(force_command))

        # Block until robot reaches target tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) <
                       self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
            # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)]
            # + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]),
            # np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2))
            # for j in range(3,6)])
            try:
                tcp_state_data = self.tcp_socket.recv(2048)
                prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            except:
                print("TCP socket Timeout!!!!")
                self.tcp_socket.close()
                self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                self.tcp_socket.settimeout(0.5)
            time.sleep(0.01)
        # self.tcp_socket.close()

    def guarded_move_to(self, tool_position, tool_orientation, speed_scale=1.0):
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')

        self.tcp_socket.send(
            str.encode(
                "force_mode(p[%f,%f,%f,%f,%f,%f], [0, 0, 1, 0, 0, 0], [0.0, 0.0, 10.0, 0.0, 0.0, 0.0], 2, [0.1, 0.1, 0.15, 0.17, 0.17, 0.17])" %
                (actual_tool_pose[0], actual_tool_pose[1], actual_tool_pose[2], actual_tool_pose[3], actual_tool_pose[4], actual_tool_pose[5])))
        # Read actual tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        execute_success = True

        # Increment every cm, check force
        self.tool_acc = 0.1  # 1.2 # 0.5
        inc_val = 0.003

        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

            # Compute motion trajectory in 1cm increments
            increment = np.asarray([(tool_position[j] - actual_tool_pose[j]) for j in range(3)])
            if np.linalg.norm(increment) < inc_val:
                increment_position = tool_position
            else:
                increment = inc_val * increment / np.linalg.norm(increment)
                increment_position = np.asarray(actual_tool_pose[0: 3]) + increment

            # Move to next increment position (blocking call)
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (increment_position[0],
                                                                               increment_position[1],
                                                                               increment_position[2],
                                                                               tool_orientation[0],
                                                                               tool_orientation[1],
                                                                               tool_orientation[2],
                                                                               self.tool_acc,
                                                                               self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            time_start = time.time()
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - increment_position[j])
                           < self.tool_pose_tolerance[j] for j in range(3)]):
                # print([np.abs(actual_tool_pose[j] - increment_position[j]) for j in range(3)])
                tcp_state_data = self.tcp_socket.recv(2048)
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                print(self.parse_tcp_state_data(tcp_state_data, 'force_mode_data'))
                # time_snapshot = time.time()
                # if time_snapshot - time_start > 1:
                #     break
                time.sleep(0.01)

            # Reading TCP forces from real-time client connection
            rtc_state_data = self.tcp_socket.recv(2048)
            forces = self.parse_tcp_state_data(rtc_state_data, "force_mode_data")

            print(forces)

            # If TCP forces in x/y exceed 20 Newtons, stop moving
            # print(TCP_forces[0:3])
            # if np.linalg.norm(np.asarray(TCP_forces[0: 2])) > 20 or (time_snapshot - time_start) > 1:
            if np.linalg.norm(np.asarray(forces[0: 2])) > 20:
                print(
                    'Warning: contact detected! Movement halted. TCP forces: [%f, %f, %f]' %
                    (forces[0], forces[1], forces[2]))
                execute_success = False
                break

            time.sleep(0.01)

        self.tool_acc = 1.2  # 1.2 # 0.5
        self.tcp_socket.send(str.encode("end_force_mode()"))
        # self.tcp_socket.close()
        # self.rtc_socket.close()

        return execute_success

    def move_joints(self, joint_configuration):

        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(2048)
        state_data = self.tcp_socket.recv(2048)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j])
                       < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(2048)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)

        # self.tcp_socket.close()

    def go_home(self):

        # self.move_to([-0.2, -0.18, 0.4], [0, -3.14, 0])
        self.joint_acc = 1.4  # Safe: 1.4
        self.joint_vel = 1.05  # Safe: 1.05
        self.move_joints(self.home_joint_config)
        self.joint_acc = 8  # Safe: 1.4
        self.joint_vel = 3  # Safe: 1.05

    # Note: must be preceded by close_gripper()

    def check_grasp(self):
        return self.get_gripper_pos() < 220
        # return True

        # state_data = self.get_state()
        # tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        # return tool_analog_input2 > 0.26

    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

            # Move gripper to location above grasp target
            grasp_location_margin = 0.15
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] -
                                         UR5_target_position[0], tool_position[1] -
                                         UR5_target_position[1], tool_position[2] -
                                         UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            if move_step[0] == 0:
                num_move_steps = 1
            else:
                num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                           1, (UR5_target_position[0] +
                                               move_step[0] *
                                               min(step_iter, num_move_steps), UR5_target_position[1] +
                                               move_step[1] *
                                               min(step_iter, num_move_steps), UR5_target_position[2] +
                                               move_step[2] *
                                               min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -
                                              1, (np.pi /
                                                  2, gripper_orientation[1] +
                                                  rotation_step *
                                                  min(step_iter, num_rotation_steps), np.pi /
                                                  2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                       1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

            # Ensure gripper is open
            self.open_gripper()

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            gripper_full_closed = self.close_gripper()

            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)

            # Check if grasp is successful
            gripper_full_closed = self.close_gripper()
            grasp_success = not gripper_full_closed

            # Move the grasped object elsewhere
            if grasp_success:
                object_positions = np.asarray(self.get_obj_positions())
                object_positions = object_positions[:, 2]
                grasped_object_ind = np.argmax(object_positions)
                grasped_object_handle = self.object_handles[grasped_object_ind]
                vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1, (-0.5,
                                                                                        0.5 + 0.05 * float(grasped_object_ind), 0.1), vrep.simx_opmode_blocking)

        else:

            # Compute tool orientation from heightmap rotation angle
            grasp_orientation = [-1.0, 0.0]
            while heightmap_rotation_angle > 2 * np.pi or heightmap_rotation_angle < 0:
                if heightmap_rotation_angle > 2 * np.pi:
                    heightmap_rotation_angle -= 2 * np.pi
                else:
                    heightmap_rotation_angle += 2 * np.pi
            if heightmap_rotation_angle <= np.pi / 2:
                heightmap_rotation_angle += np.pi
            if heightmap_rotation_angle >= np.pi * 3 / 2:
                heightmap_rotation_angle -= np.pi
            # if heightmap_rotation_angle > np.pi:
            #     heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
            tool_rotation_angle = heightmap_rotation_angle / 2
            tool_orientation = np.asarray(
                [
                    grasp_orientation[0] * np.cos(tool_rotation_angle) -
                    grasp_orientation[1] * np.sin(tool_rotation_angle),
                    grasp_orientation[0] * np.sin(tool_rotation_angle) +
                    grasp_orientation[1] * np.cos(tool_rotation_angle),
                    0.0]) * np.pi
            tool_orientation_angle = np.linalg.norm(tool_orientation)
            tool_orientation_axis = tool_orientation / tool_orientation_angle
            tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

            # Compute tilted tool orientation during dropping into bin
            tilt_rotm = utils.euler2rotm(np.asarray([-np.pi / 4, 0, 0]))
            tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
            tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
            tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * \
                np.asarray(tilted_tool_orientation_axis_angle[1:4])

            # Attempt grasp
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.001)
            self.open_gripper()
            self.move_to([position[0], position[1], position[2] + 0.1],
                         [tool_orientation[0], tool_orientation[1], 0.0], 0.5)
            move_success = self.protected_move_to([position[0], position[1], position[2]], [
                tool_orientation[0], tool_orientation[1], 0.0], 0.1)
            grasp_success = False
            if move_success:
                self.close_gripper()
                # bin_position = [-0.294, -0.465, 0.4]
                bin_position = [-0.32, 0, 0.5]
                # If gripper is open, drop object in bin and check if grasp is successful

                self.move_to([position[0], position[1], position[2] + 0.1],
                             [tool_orientation[0], tool_orientation[1], 0.0], 0.5)
                self.close_gripper()

                grasp_success = self.check_grasp()
                # grasp_success = int(input("Successfully grasped? "))

                if grasp_success:
                    self.move_to([position[0], position[1], bin_position[2]], [
                        tool_orientation[0], tool_orientation[1], 0.0])
                    self.move_to([bin_position[0], bin_position[1], bin_position[2]], [
                        tool_orientation[0], tool_orientation[1], 0.0])
                    self.close_gripper()
                    grasp_success = self.check_grasp()
                    self.open_gripper()
            else:
                self.move_to([position[0], position[1], position[2] + 0.1],
                             [tool_orientation[0], tool_orientation[1], 0.0], 0.5)
            self.go_home()
        return grasp_success

    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f) with %f' %
              (position[0], position[1], position[2], heightmap_rotation_angle))
        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi)

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026

            # Compute pushing direction
            push_orientation = [1.0, 0.0]
            push_direction = np.asarray([push_orientation[0] *
                                         np.cos(heightmap_rotation_angle) -
                                         push_orientation[1] *
                                         np.sin(heightmap_rotation_angle), push_orientation[0] *
                                         np.sin(heightmap_rotation_angle) +
                                         push_orientation[1] *
                                         np.cos(heightmap_rotation_angle)])

            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] -
                                         UR5_target_position[0], tool_position[1] -
                                         UR5_target_position[1], tool_position[2] -
                                         UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            if move_step[0] == 0:
                num_move_steps = 2
            else:
                num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                           1, (UR5_target_position[0] +
                                               move_step[0] *
                                               min(step_iter, num_move_steps), UR5_target_position[1] +
                                               move_step[1] *
                                               min(step_iter, num_move_steps), UR5_target_position[2] +
                                               move_step[2] *
                                               min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -
                                              1, (np.pi /
                                                  2, gripper_orientation[1] +
                                                  rotation_step *
                                                  min(step_iter, num_rotation_steps), np.pi /
                                                  2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                       1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            # Compute target location (push to the right)
            push_length = 0.1
            target_x = min(max(position[0] + push_direction[0] * push_length,
                               workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1] * push_length,
                               workspace_limits[1][0]), workspace_limits[1][1])
            push_length = np.sqrt(np.power(target_x - position[0], 2) + np.power(target_y - position[1], 2))

            # Move in pushing direction towards target location
            self.move_to([target_x, target_y, position[2]], None)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            push_success = True

        else:

            # Compute tool orientation from heightmap rotation angle
            push_orientation = [1.0, 0.0]
            # if heightmap_rotation_angle <= np.pi / 2:
            #     heightmap_rotation_angle += np.pi
            # if heightmap_rotation_angle >= np.pi * 3 / 2:
            #     heightmap_rotation_angle -= np.pi
            tool_rotation_angle = heightmap_rotation_angle / 2 + np.pi / 4
            tool_orientation = np.asarray(
                [
                    push_orientation[0] * np.cos(tool_rotation_angle) -
                    push_orientation[1] * np.sin(tool_rotation_angle),
                    push_orientation[0] * np.sin(tool_rotation_angle) +
                    push_orientation[1] * np.cos(tool_rotation_angle),
                    0.0]) * np.pi
            tool_orientation_angle = np.linalg.norm(tool_orientation)
            tool_orientation_axis = tool_orientation / tool_orientation_angle
            tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

            # Compute push direction and endpoint (push to right of rotated heightmap)
            push_direction = np.asarray([push_orientation[0] *
                                         np.cos(heightmap_rotation_angle) -
                                         push_orientation[1] *
                                         np.sin(heightmap_rotation_angle), push_orientation[0] *
                                         np.sin(heightmap_rotation_angle) +
                                         push_orientation[1] *
                                         np.cos(heightmap_rotation_angle), 0.0])
            target_x = min(max(position[0] + push_direction[0] * 0.1, workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1] * 0.1, workspace_limits[1][0]), workspace_limits[1][1])
            push_endpoint = np.asarray([target_x, target_y, position[2]])
            push_direction.shape = (3, 1)

            # Compute tilted tool orientation during push
            tilt_axis = np.dot(utils.euler2rotm(np.asarray([0, 0, np.pi / 2]))[:3, :3], push_direction)
            tilt_rotm = utils.angle2rotm(-np.pi / 8, tilt_axis, point=None)[:3, :3]
            tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
            tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
            tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * \
                np.asarray(tilted_tool_orientation_axis_angle[1:4])

            # Push only within workspace limits
            position = np.asarray(position).copy()
            position[0] = min(max(position[0], workspace_limits[0][0]), workspace_limits[0][1])
            position[1] = min(max(position[1], workspace_limits[1][0]), workspace_limits[1][1])
            position[2] = max(position[2] + 0.005, workspace_limits[2][0] + 0.005)  # Add buffer to surface

            self.close_gripper()
            self.move_to([position[0],
                          position[1],
                          position[2] + 0.1],
                         [tool_orientation[0],
                          tool_orientation[1],
                          tool_orientation[2]],
                         1.0)
            down_success = self.protected_move_to([position[0],
                                                   position[1],
                                                   position[2]],
                                                  [tool_orientation[0],
                                                   tool_orientation[1],
                                                   tool_orientation[2]],
                                                  0.1)
            if down_success:
                print("Pushing...")
                self.protected_move_to([push_endpoint[0],
                                        push_endpoint[1],
                                        position[2]],
                                       [tool_orientation[0],
                                        tool_orientation[1],
                                        tool_orientation[2]],
                                       0.1, 80)
            self.move_to([position[0],
                          position[1],
                          position[2] + 0.1],
                         [tool_orientation[0],
                             tool_orientation[1],
                             tool_orientation[2]],
                         1.0)
            self.go_home()

            push_success = True
            time.sleep(0.1)

        return push_success

    def push_for_measuring(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026

            # Compute pushing direction
            push_orientation = [1.0, 0.0]
            push_direction = np.asarray([push_orientation[0] *
                                         np.cos(heightmap_rotation_angle) -
                                         push_orientation[1] *
                                         np.sin(heightmap_rotation_angle), push_orientation[0] *
                                         np.sin(heightmap_rotation_angle) +
                                         push_orientation[1] *
                                         np.cos(heightmap_rotation_angle)])

            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] -
                                         UR5_target_position[0], tool_position[1] -
                                         UR5_target_position[1], tool_position[2] -
                                         UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            if move_step[0] == 0:
                num_move_steps = 2
            else:
                num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                           1, (UR5_target_position[0] +
                                               move_step[0] *
                                               min(step_iter, num_move_steps), UR5_target_position[1] +
                                               move_step[1] *
                                               min(step_iter, num_move_steps), UR5_target_position[2] +
                                               move_step[2] *
                                               min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -
                                              1, (np.pi /
                                                  2, gripper_orientation[1] +
                                                  rotation_step *
                                                  min(step_iter, num_rotation_steps), np.pi /
                                                  2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                       1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            color, depth = self.get_camera_data()
            color_images = [color]
            depth_images = [depth]
            actions = [position[:2]]

            next_obj_positions, next_obj_orientations = self.get_obj_positions_and_orientations()
            pose = []
            for i in range(len(next_obj_positions)):
                pose.append(next_obj_positions[i][0])
                pose.append(next_obj_positions[i][1])
                pose.append(next_obj_orientations[i][0])
                pose.append(next_obj_orientations[i][1])
                pose.append(next_obj_orientations[i][2])
            pose = tuple(pose)
            poses = [pose]

            # Compute target location (push to the right)
            for i in range(2):
                push_length = i * 0.05 + 0.05
                target_x = min(max(position[0] + push_direction[0] * push_length,
                                   workspace_limits[0][0]), workspace_limits[0][1])
                target_y = min(max(position[1] + push_direction[1] * push_length,
                                   workspace_limits[1][0]), workspace_limits[1][1])

                # Move in pushing direction towards target location
                self.move_to([target_x, target_y, position[2]], None)

                color, depth = self.get_camera_data()
                color_images.append(color)
                depth_images.append(depth)
                actions.append((target_x, target_y))
                next_obj_positions, next_obj_orientations = self.get_obj_positions_and_orientations()
                pose_tuples = []
                for i in range(len(next_obj_positions)):
                    pose_tuples.append(next_obj_positions[i][0])
                    pose_tuples.append(next_obj_positions[i][1])
                    pose_tuples.append(next_obj_orientations[i][0])
                    pose_tuples.append(next_obj_orientations[i][1])
                    pose_tuples.append(next_obj_orientations[i][2])
                pose_tuples = tuple(pose_tuples)
                poses.append(pose_tuples)
            actions[-1] = (0, 0)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            push_success = True

        else:
            pass

        return push_success, color_images, depth_images, actions, poses

    def push_with_stream(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi)

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026

            # Compute pushing direction
            push_orientation = [1.0, 0.0]
            push_direction = np.asarray([push_orientation[0] *
                                         np.cos(heightmap_rotation_angle) -
                                         push_orientation[1] *
                                         np.sin(heightmap_rotation_angle), push_orientation[0] *
                                         np.sin(heightmap_rotation_angle) +
                                         push_orientation[1] *
                                         np.cos(heightmap_rotation_angle)])

            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] -
                                         UR5_target_position[0], tool_position[1] -
                                         UR5_target_position[1], tool_position[2] -
                                         UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05 * move_direction / move_magnitude
            if move_step[0] == 0:
                num_move_steps = 2
            else:
                num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(
                self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                           1, (UR5_target_position[0] +
                                               move_step[0] *
                                               min(step_iter, num_move_steps), UR5_target_position[1] +
                                               move_step[1] *
                                               min(step_iter, num_move_steps), UR5_target_position[2] +
                                               move_step[2] *
                                               min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -
                                              1, (np.pi /
                                                  2, gripper_orientation[1] +
                                                  rotation_step *
                                                  min(step_iter, num_rotation_steps), np.pi /
                                                  2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -
                                       1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            color, depth = self.get_camera_data()
            color_images = [color]
            depth_images = [depth]
            actions = [position[:2]]

            next_obj_positions, next_obj_orientations = self.get_obj_positions_and_orientations()
            pose = []
            for i in range(len(next_obj_positions)):
                pose.append(next_obj_positions[i][0])
                pose.append(next_obj_positions[i][1])
                pose.append(next_obj_orientations[i][0])
                pose.append(next_obj_orientations[i][1])
                pose.append(next_obj_orientations[i][2])
            pose = tuple(pose)
            poses = [pose]

            # Compute target location (push to the right)
            for i in range(10):
                push_length = i * 0.01 + 0.01
                target_x = min(max(position[0] + push_direction[0] * push_length,
                                   workspace_limits[0][0]), workspace_limits[0][1])
                target_y = min(max(position[1] + push_direction[1] * push_length,
                                   workspace_limits[1][0]), workspace_limits[1][1])

                # Move in pushing direction towards target location
                self.move_to([target_x, target_y, position[2]], None)
                time.sleep(0.2)

                color, depth = self.get_camera_data()
                color_images.append(color)
                depth_images.append(depth)
                actions.append((target_x, target_y))
                next_obj_positions, next_obj_orientations = self.get_obj_positions_and_orientations()
                pose_tuples = []
                for i in range(len(next_obj_positions)):
                    pose_tuples.append(next_obj_positions[i][0])
                    pose_tuples.append(next_obj_positions[i][1])
                    pose_tuples.append(next_obj_orientations[i][0])
                    pose_tuples.append(next_obj_orientations[i][1])
                    pose_tuples.append(next_obj_orientations[i][2])
                pose_tuples = tuple(pose_tuples)
                poses.append(pose_tuples)
            actions[-1] = (0, 0)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            push_success = True

        else:
            # Compute tool orientation from heightmap rotation angle
            push_orientation = [1.0, 0.0]
            # if heightmap_rotation_angle <= np.pi / 2:
            #     heightmap_rotation_angle += np.pi
            # if heightmap_rotation_angle >= np.pi * 3 / 2:
            #     heightmap_rotation_angle -= np.pi
            tool_rotation_angle = heightmap_rotation_angle / 2 + np.pi / 4
            tool_orientation = np.asarray(
                [
                    push_orientation[0] * np.cos(tool_rotation_angle) -
                    push_orientation[1] * np.sin(tool_rotation_angle),
                    push_orientation[0] * np.sin(tool_rotation_angle) +
                    push_orientation[1] * np.cos(tool_rotation_angle),
                    0.0]) * np.pi
            tool_orientation_angle = np.linalg.norm(tool_orientation)
            tool_orientation_axis = tool_orientation / tool_orientation_angle
            tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

            # Compute push direction and endpoint (push to right of rotated heightmap)
            push_direction = np.asarray([push_orientation[0] *
                                         np.cos(heightmap_rotation_angle) -
                                         push_orientation[1] *
                                         np.sin(heightmap_rotation_angle), push_orientation[0] *
                                         np.sin(heightmap_rotation_angle) +
                                         push_orientation[1] *
                                         np.cos(heightmap_rotation_angle), 0.0])
            target_x = min(max(position[0] + push_direction[0] * 0.1, workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1] * 0.1, workspace_limits[1][0]), workspace_limits[1][1])
            push_endpoint = np.asarray([target_x, target_y, position[2]])
            push_direction.shape = (3, 1)

            # Compute tilted tool orientation during push
            tilt_axis = np.dot(utils.euler2rotm(np.asarray([0, 0, np.pi / 2]))[:3, :3], push_direction)
            tilt_rotm = utils.angle2rotm(-np.pi / 8, tilt_axis, point=None)[:3, :3]
            tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
            tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
            tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * \
                np.asarray(tilted_tool_orientation_axis_angle[1:4])

            # Push only within workspace limits
            position = np.asarray(position).copy()
            position[0] = min(max(position[0], workspace_limits[0][0]), workspace_limits[0][1])
            position[1] = min(max(position[1], workspace_limits[1][0]), workspace_limits[1][1])
            position[2] = max(position[2] + 0.005, workspace_limits[2][0] + 0.005)  # Add buffer to surface

            # # record the first image
            # color, depth = self.get_camera_data()
            # color_images = [color]
            # depth_images = [depth]
            # actions = [position[:2]]
            # # get positions and orientations of these objects
            # next_obj_positions, next_obj_orientations = self.get_real_location()
            # pose = []
            # for i in range(len(next_obj_positions)):
            #     pose.append(next_obj_positions[i][0])
            #     pose.append(next_obj_positions[i][1])
            #     pose.append(next_obj_orientations[i][0])
            #     pose.append(next_obj_orientations[i][1])
            #     pose.append(next_obj_orientations[i][2])
            # pose = tuple(pose)
            # poses = [pose]


            self.close_gripper()
            self.move_to([position[0],
                          position[1],
                          position[2] + 0.1],
                         [tool_orientation[0],
                          tool_orientation[1],
                          tool_orientation[2]],
                         1.0)
            down_success = self.protected_move_to([position[0],
                                                   position[1],
                                                   position[2]],
                                                  [tool_orientation[0],
                                                   tool_orientation[1],
                                                   tool_orientation[2]],
                                                  0.1)
            if down_success:
                print("Pushing...")
                self.protected_move_to([push_endpoint[0],
                                        push_endpoint[1],
                                        position[2]],
                                       [tool_orientation[0],
                                        tool_orientation[1],
                                        tool_orientation[2]],
                                       0.1, 80)
            self.move_to([position[0],
                          position[1],
                          position[2] + 0.1],
                         [tool_orientation[0],
                             tool_orientation[1],
                             tool_orientation[2]],
                         1.0)
            self.go_home()

            # # record the last image
            # color, depth = self.get_camera_data()
            # color_images.append(color)
            # depth_images.append(depth)
            # actions.append((0, 0))
            # next_obj_positions, next_obj_orientations = self.get_real_location()
            # pose_tuples = []
            # for i in range(len(next_obj_positions)):
            #     pose_tuples.append(next_obj_positions[i][0])
            #     pose_tuples.append(next_obj_positions[i][1])
            #     pose_tuples.append(next_obj_orientations[i][0])
            #     pose_tuples.append(next_obj_orientations[i][1])
            #     pose_tuples.append(next_obj_orientations[i][2])
            # pose_tuples = tuple(pose_tuples)
            # poses.append(pose_tuples)

            push_success = True
            time.sleep(0.1)

        return push_success, color_images, depth_images, actions, poses

    def get_real_location(self):
        pass

    def restart_real(self):

        # TODO flip box, reallocate objects

        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0, 0.0]
        tool_rotation_angle = -np.pi / 4
        tool_orientation = np.asarray(
            [
                grasp_orientation[0] * np.cos(tool_rotation_angle) - grasp_orientation[1] * np.sin(tool_rotation_angle),
                grasp_orientation[0] * np.sin(tool_rotation_angle) + grasp_orientation[1] * np.cos(tool_rotation_angle),
                0.0]) * np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation / tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

        tilt_rotm = utils.euler2rotm(np.asarray([-np.pi / 4, 0, 0]))
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * \
            np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # Move to box grabbing position
        box_grab_position = [0.5, -0.35, -0.12]
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],
                                                                                box_grab_position[1],
                                                                                box_grab_position[2] + 0.1,
                                                                                tilted_tool_orientation[0],
                                                                                tilted_tool_orientation[1],
                                                                                tilted_tool_orientation[2],
                                                                                self.joint_acc,
                                                                                self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],
                                                                                box_grab_position[1],
                                                                                box_grab_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc,
                                                                                self.joint_vel)
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        # self.tcp_socket.close()

        # Block until robot reaches box grabbing position and gripper fingers have stopped moving
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all(
                    [np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

        # Move to box release position
        box_release_position = [0.5, 0.08, -0.12]
        home_position = [0.49, 0.11, 0.03]
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],
                                                                                box_release_position[1],
                                                                                box_release_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.1,
                                                                                self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],
                                                                                box_release_position[1],
                                                                                box_release_position[2] + 0.3,
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.02,
                                                                                self.joint_vel * 0.02)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (box_grab_position[0] - 0.05,
                                                                                box_grab_position[1] + 0.1,
                                                                                box_grab_position[2] + 0.3,
                                                                                tilted_tool_orientation[0],
                                                                                tilted_tool_orientation[1],
                                                                                tilted_tool_orientation[2],
                                                                                self.joint_acc * 0.5,
                                                                                self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0] - 0.05,
                                                                                box_grab_position[1] + 0.1,
                                                                                box_grab_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.5,
                                                                                self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],
                                                                                box_grab_position[1],
                                                                                box_grab_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.1,
                                                                                self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0] + 0.05,
                                                                                box_grab_position[1],
                                                                                box_grab_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc * 0.1,
                                                                                self.joint_vel * 0.1)
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],
                                                                                box_grab_position[1],
                                                                                box_grab_position[2] + 0.1,
                                                                                tilted_tool_orientation[0],
                                                                                tilted_tool_orientation[1],
                                                                                tilted_tool_orientation[2],
                                                                                self.joint_acc,
                                                                                self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],
                                                                                home_position[1],
                                                                                home_position[2],
                                                                                tool_orientation[0],
                                                                                tool_orientation[1],
                                                                                tool_orientation[2],
                                                                                self.joint_acc,
                                                                                self.joint_vel)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        # self.tcp_socket.close()

        # Block until robot reaches home position
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all(
                    [np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2
