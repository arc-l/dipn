from robot import Robot
import utils
import os
import time
import threading
import datetime
import argparse
import numpy as np
import cv2
from constants import workspace_limits, heightmap_resolution, DEPTH_MIN

class PushDataCollector():

    def __init__(self, args):
        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.continue_logging = args.continue_logging
        if self.continue_logging:
            self.base_directory = os.path.abspath(args.logging_directory)
            print('Pre-loading data logging session: %s' % (self.base_directory))
        else:
            self.base_directory = os.path.join(os.path.abspath('logs_push'), timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            print('Creating data logging session: %s' % (self.base_directory))
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        # self.prev_color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'prev_color_heightmaps')
        # self.prev_depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'prev_depth_heightmaps')
        # self.prev_pose_heightmaps_directory = os.path.join(self.base_directory, 'data', 'prev_pose')
        # self.next_color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'next_color_heightmaps')
        # self.next_depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'next_depth_heightmaps')
        # self.next_pose_heightmaps_directory = os.path.join(self.base_directory, 'data', 'next_pose')
        self.action_directory = os.path.join(self.base_directory, 'data', 'actions')
        self.pose_heightmaps_directory = os.path.join(self.base_directory, 'data', 'poses')
        
        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        # if not os.path.exists(self.prev_color_heightmaps_directory):
        #     os.makedirs(self.prev_color_heightmaps_directory)
        # if not os.path.exists(self.prev_depth_heightmaps_directory):
        #     os.makedirs(self.prev_depth_heightmaps_directory)
        # if not os.path.exists(self.prev_pose_heightmaps_directory):
        #     os.makedirs(self.prev_pose_heightmaps_directory)
        # if not os.path.exists(self.next_color_heightmaps_directory):
        #     os.makedirs(self.next_color_heightmaps_directory)
        # if not os.path.exists(self.next_depth_heightmaps_directory):
        #     os.makedirs(self.next_depth_heightmaps_directory)
        # if not os.path.exists(self.next_pose_heightmaps_directory):
        #     os.makedirs(self.next_pose_heightmaps_directory)
        if not os.path.exists(self.action_directory):
            os.makedirs(self.action_directory)
        if not os.path.exists(self.pose_heightmaps_directory):
            os.makedirs(self.pose_heightmaps_directory)

        self.iter = args.start_iter
        self.start_iter = args.start_iter
        self.end_iter = args.end_iter

        self.loaded = False
        self.saving_color_images = None
        self.saving_depth_images = None
        self.saving_actions = None
        self.saving_poses = None
        self.saving_iter = self.iter

    def save_push_prediction_heightmaps(self, iteration, prev_color_heightmap, prev_depth_heightmap, next_color_heightmap, next_depth_heightmap):
        color_heightmap = cv2.cvtColor(prev_color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.prev_color_heightmaps_directory, '%07d.color.png' % (iteration)), color_heightmap)
        depth_heightmap = np.round(prev_depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.prev_depth_heightmaps_directory, '%07d.depth.png' % (iteration)), depth_heightmap)

        color_heightmap = cv2.cvtColor(next_color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.next_color_heightmaps_directory, '%07d.color.png' % (iteration)), color_heightmap)
        depth_heightmap = np.round(next_depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.next_depth_heightmaps_directory, '%07d.depth.png' % (iteration)), depth_heightmap)

    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, '%07d.color.png' % (iteration)), color_heightmap)
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.depth_heightmaps_directory, '%07d.depth.png' % (iteration)), depth_heightmap)

    def save_action(self, iteration, pose):
        np.savetxt(os.path.join(self.action_directory, '%07d.action.txt' % (iteration)), pose, fmt='%s')

    def save_pose(self, iteration, pose):
        np.savetxt(os.path.join(self.pose_heightmaps_directory, '%07d.pose.txt' % (iteration)), pose, fmt='%s')

    def push_check(self, args):
        """
        Script to check the correctness of collection process
        """
        # --------------- Setup options ---------------
        is_sim = args.is_sim # Run in simulation?
        obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
        num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
        tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
        tcp_port = args.tcp_port if not is_sim else None
        rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
        rtc_port = args.rtc_port if not is_sim else None
        
        # Initialize pick-and-place system (camera and robot)
        robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                    tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                    is_testing=False, test_preset_cases=False, test_preset_file='', collect_push=True)

        print('\nStart', self.iter)

        robot.check_sim()
        robot.restart_sim()
        robot.add_object_push()
        start_x, start_y = input("Input action position: ").split()
        start_x = float(start_x)
        start_y = float(start_y)

        for i in range(10):
            # Get latest RGB-D image
            color_img, depth_img = robot.get_camera_data()
            depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
            prev_color_heightmap, prev_depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
            prev_valid_depth_heightmap = prev_depth_heightmap.copy()
            prev_valid_depth_heightmap[np.isnan(prev_valid_depth_heightmap)] = 0
            prev_obj_positions, prev_obj_orientations = robot.get_obj_positions_and_orientations()
            poses = (prev_obj_positions[0][0], prev_obj_positions[0][1], prev_obj_orientations[0][0], prev_obj_orientations[0][1], prev_obj_orientations[0][2], 
                        prev_obj_positions[1][0], prev_obj_positions[1][1], prev_obj_orientations[1][0], prev_obj_orientations[1][1], prev_obj_orientations[1][2])
            print(prev_obj_positions[0], prev_obj_orientations[0])
            print(prev_obj_positions[1], prev_obj_orientations[1])

            # push 1 cm
            action = [start_x + i * 0.01, start_y, 0.001]
            push_success = robot.push(action, 0, workspace_limits)
            assert push_success
            
            input("press to continue")

            self.save_heightmaps(self.iter * 100 + i, prev_color_heightmap, prev_valid_depth_heightmap)
            self.save_action(self.iter * 100 + i, [action[:2]])
            self.save_pose(self.iter * 100 + i, [poses])

         # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        prev_color_heightmap, prev_depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        prev_valid_depth_heightmap = prev_depth_heightmap.copy()
        prev_valid_depth_heightmap[np.isnan(prev_valid_depth_heightmap)] = 0
        prev_obj_positions, prev_obj_orientations = robot.get_obj_positions_and_orientations()
        poses = (prev_obj_positions[0][0], prev_obj_positions[0][1], prev_obj_orientations[0][0], prev_obj_orientations[0][1], prev_obj_orientations[0][2], 
                    prev_obj_positions[1][0], prev_obj_positions[1][1], prev_obj_orientations[1][0], prev_obj_orientations[1][1], prev_obj_orientations[1][2])
        print(prev_obj_positions[0], prev_obj_orientations[0])
        print(prev_obj_positions[1], prev_obj_orientations[1])
        self.save_heightmaps(self.iter * 100 + i + 1 , prev_color_heightmap, prev_valid_depth_heightmap)
        self.save_pose(self.iter * 100 + i + 1, [poses])
        
    def push_data_collect(self, args):
        """
        Randomly dropped objects to the workspace, the robot makes a push from left to right, recording the info every 1 cm.
        """
        # --------------- Setup options ---------------
        is_sim = args.is_sim # Run in simulation?
        obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
        num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
        tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
        tcp_port = args.tcp_port if not is_sim else None
        rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
        rtc_port = args.rtc_port if not is_sim else None
        # -------------- Testing options --------------
        is_testing = args.is_testing
        test_preset_cases = args.test_preset_cases
        test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None
        
        # Initialize pick-and-place system (camera and robot)
        robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                    tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                    is_testing=is_testing, test_preset_cases=test_preset_cases, test_preset_file=test_preset_file, collect_push=True)

        thread = threading.Thread(target=self.saving_thread, args=(robot,))
        thread.start()

        while self.iter < self.end_iter:
            print('\nCollecting data iteration: %d' % (self.iter))

            # Make sure simulation is still stable (if not, reset simulation)
            if is_sim: 
                robot.check_sim()
                robot.restart_sim()
                bbox_heights = robot.add_object_push()

            # Get latest RGB-D image
            color_img, depth_img = robot.get_camera_data()
            depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
            prev_color_heightmap, prev_depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
            prev_valid_depth_heightmap = prev_depth_heightmap.copy()
            prev_valid_depth_heightmap[np.isnan(prev_valid_depth_heightmap)] = 0
            prev_obj_positions, prev_obj_orientations = robot.get_obj_positions_and_orientations()
            skip = False
            for i in range(len(prev_obj_positions)):
                if (prev_obj_positions[i][0] < workspace_limits[0][0] or prev_obj_positions[i][0] > workspace_limits[0][1] or 
                    prev_obj_positions[i][1] < workspace_limits[1][0] or prev_obj_positions[i][1] > workspace_limits[1][1]):
                    print("Out of space, Skip")
                    skip = True
                    break
                if prev_obj_positions[i][2] > bbox_heights[i]:
                    print("height is wrong Skip")
                    skip = True
                    break
            if skip: continue

            # Find target and push
            depth_heightmap = np.copy(prev_valid_depth_heightmap)
            depth_heightmap[depth_heightmap <= DEPTH_MIN] = 0
            depth_heightmap[depth_heightmap > DEPTH_MIN] = 1

            y_indices = np.argwhere(depth_heightmap == 1)[:, 0]  # Find the y range
            if len(y_indices) == 0: 
                print("find Skip")
                continue
            y_list = np.arange(y_indices.min(), y_indices.max() + 1)
            if len(y_list) == 0:
                print("min Skip")
                continue
            y_list = y_list[10:len(y_list)-10]
            if len(y_list) == 0:
                print("shrink Skip")
                continue
            y = np.random.choice(y_list)
            x_indices = np.argwhere(depth_heightmap[y, :] == 1)[:, 0]  # Find the x range
            x_indices_up = np.argwhere(depth_heightmap[max(y-5, 0)] == 1)[:, 0]  # Find the x range
            x_indices_down = np.argwhere(depth_heightmap[min(y+5, 223)] == 1)[:, 0]  # Find the x range
            if len(x_indices) == 0:
                print("Skip")
                continue
            x = x_indices.min()
            if len(x_indices_up) != 0:
                x = min(x, x_indices_up.min())
            if len(x_indices_down) != 0:
                x = min(x, x_indices_down.min())
            x =  x - 10
            if x <= 0:
                print("Skip")
                continue

            # safe_kernel = 16
            # local_region = prev_valid_depth_heightmap[max(x - safe_kernel, 0):min(y + safe_kernel + 1, prev_valid_depth_heightmap.shape[0]), max(x - safe_kernel, 0):min(x + safe_kernel + 1, prev_valid_depth_heightmap.shape[1])]
            # if local_region.size == 0:
            #     safe_z_position = workspace_limits[2][0]
            # else:
            #     if np.max(local_region) < 0.03:
            #         safe_z_position = workspace_limits[2][0]
            #     else:
            #         safe_z_position = 0.025 + workspace_limits[2][0]
            safe_z_position = workspace_limits[2][0]

            _, color_images, depth_images, actions, poses = robot.push_with_stream([x * heightmap_resolution + workspace_limits[0][0], y * heightmap_resolution + workspace_limits[1][0], safe_z_position], 0, workspace_limits)
            while self.loaded:
                print('Wait for saving iteration:', self.saving_iter)
                time.sleep(1)
            self.saving_color_images = color_images
            self.saving_depth_images = depth_images
            self.saving_actions = actions
            self.saving_poses = poses
            self.saving_iter = self.iter
            self.loaded = True
                
            self.iter += 1

    def push_data_collect_real(self, args):
        """
        Becase sim to real is working, this part hasn't been completed, but the idea and process should be the same
        """
        # --------------- Setup options ---------------
        is_sim = args.is_sim  # Run in simulation?
        # Directory containing 3D mesh files (.obj) of objects to be added to simulation
        obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None
        num_obj = args.num_obj if is_sim else None  # Number of objects to add to simulation
        tcp_host_ip = '172.19.97.157' # IP and port to robot arm as TCP client (UR5)
        tcp_port = 30002
        rtc_host_ip = '172.19.97.157'  # IP and port to robot arm as real-time client (UR5)
        rtc_port = 30003
        # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        heightmap_resolution = args.heightmap_resolution  # Meters per pixel of heightmap
        is_testing = args.is_testing
        test_preset_cases = args.test_preset_cases
        test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None
        
        # Initialize pick-and-place system (camera and robot)
        robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                    tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                    is_testing=is_testing, test_preset_cases=test_preset_cases, test_preset_file=test_preset_file, collect_push=True)

        thread = threading.Thread(target=self.saving_thread, args=(robot,))
        thread.start()

        while self.iter < self.end_iter:
            print('\nCollecting data iteration: %d' % (self.iter))

            # Get latest RGB-D image
            color_img, depth_img = robot.get_camera_data()

            # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
            prev_color_heightmap, prev_depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
            prev_valid_depth_heightmap = prev_depth_heightmap.copy()
            prev_valid_depth_heightmap[np.isnan(prev_valid_depth_heightmap)] = 0

            # center
            x = 40
            y = 112
            safe_z_position = workspace_limits[2][0] + 0.1

            _, color_images, depth_images, actions, poses = robot.push_with_stream([x * heightmap_resolution + workspace_limits[0][0], y * heightmap_resolution + workspace_limits[1][0], safe_z_position], 0, workspace_limits)
            while self.loaded:
                print('Wait for saving iteration:', self.saving_iter)
                time.sleep(1)
            self.saving_color_images = color_images
            self.saving_depth_images = depth_images
            self.saving_actions = actions
            self.saving_poses = poses
            self.saving_iter = self.iter
            self.loaded = True
                
            self.iter += 1

    def saving_thread(self, robot):
        print('Saving started')
        while True:
            if self.loaded:
                print('Saving iteration:', self.saving_iter)
                for i in range(len(self.saving_color_images)):
                    color_img = self.saving_color_images[i]
                    depth_img = self.saving_depth_images[i]
                    depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

                    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
                    next_color_heightmap, next_depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
                    next_valid_depth_heightmap = next_depth_heightmap.copy()
                    next_valid_depth_heightmap[np.isnan(next_valid_depth_heightmap)] = 0

                    self.save_heightmaps(self.saving_iter * 100 + i, next_color_heightmap, next_valid_depth_heightmap)
                    self.save_action(self.saving_iter * 100 + i, [self.saving_actions[i]])
                    self.save_pose(self.saving_iter * 100 + i, [self.saving_poses[i]])
                    # print('Push', self.saving_actions[i])
                    # print('Pose', self.saving_poses[i])
                print('Saved iteration:', self.saving_iter)
                self.loaded = False
            else:
                time.sleep(1)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Collect data for push prediction')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--is_real', dest='is_real', action='store_true', default=False,                                    help='run in simulation?')

    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/final-push',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=7,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--check', dest='check', action='store_true', default=False)
    parser.add_argument('--start_iter', dest='start_iter', type=int, action='store', default=0)
    parser.add_argument('--end_iter', dest='end_iter', type=int, action='store', default=50000)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='new-01.txt')

    # Run main program with specified arguments
    args = parser.parse_args()

    collector = PushDataCollector(args)
    if args.check:
        collector.push_check(args)
    elif args.is_real:
        collector.push_data_collect_real(args)
    else:
        collector.push_data_collect(args)
