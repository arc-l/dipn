import cv2
import numpy as np
import imutils
from robot import Robot
from logger import Logger
from simulation import sim as vrep
import time
import os
import argparse
from constants import is_real, gripper_width, DEPTH_MIN, workspace_limits
import utils


def main(args):
    """
    Create objects in the simulation, recording images.
    If need Mask R-CNN, --is_mask should be used, which will change color of all objects to black,
      same as background. Then, change color back, take a snapshot, change to black again.
    """
    # --------------- Setup options ---------------
    is_sim = args.is_sim  # Run in simulation?
    # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None
    num_obj = args.num_obj if is_sim else None  # Number of objects to add to simulation
    tcp_host_ip = args.tcp_host_ip if not is_sim else None  # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None  # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution  # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    method = args.method  # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    # Use immediate rewards (from change detection) for pushing?
    push_rewards = args.push_rewards if method == 'reinforcement' else None
    future_reward_discount = args.future_reward_discount
    experience_replay = args.experience_replay  # Use prioritized experience replay?
    # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    heuristic_bootstrap = args.heuristic_bootstrap
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only

    # -------------- Testing options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot  # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True
    save_visualizations = args.save_visualizations

    # Set random seed
    np.random.seed(random_seed)

    hard_cases = list(sorted(os.listdir('hard-cases')))
    print('total files', len(hard_cases))

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, os.path.join('hard-cases', hard_cases[0]))

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(
        robot.cam_intrinsics,
        robot.cam_pose,
        robot.cam_depth_scale)  # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    iteration = 0

    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', iteration))

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim:
            robot.check_sim()

        time.sleep(1)
        # vrep.simxPauseSimulation(robot.sim_client, vrep.simx_opmode_blocking)         
        for idx, handle in enumerate(robot.object_handles):
            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                robot.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'staticObject', [
                    handle], [], [], bytearray(), vrep.simx_opmode_blocking)
        sim_ret, plane_handle = vrep.simxGetObjectHandle(
            robot.sim_client, 'Box', vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(robot.sim_client, plane_handle, -1,
                                               [-10, -10, 0.5], vrep.simx_opmode_blocking)

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(
            color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        valid_depth_heightmap = valid_depth_heightmap.astype(np.float32)

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(iteration, color_img, depth_img, '0')
        logger.save_heightmaps(iteration, color_heightmap, valid_depth_heightmap, '0')

        # Save Masks of each object
        if args.is_mask:
            mask = np.zeros((224, 224))
            bakcup_color = []
            # black out
            for idx, handle in enumerate(robot.object_handles):
                ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                    robot.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'changeColor', [
                        handle], [0, 0, 0], [], bytearray(), vrep.simx_opmode_blocking)
                bakcup_color.append(ret_floats)
            # record one mask at a time
            for idx, handle in enumerate(robot.object_handles):
                ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                                robot.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'changeColor', [
                                    handle], bakcup_color[idx], [], bytearray(), vrep.simx_opmode_blocking)

                color_img, depth_img = robot.get_camera_data()
                color_heightmap, depth_heightmap = utils.get_heightmap(
                    color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
                gray = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2GRAY)
                gray = gray.astype(np.uint8)
                blurred = cv2.medianBlur(gray, 5)
                thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
                mask[thresh == 255] = idx + 1

                ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                    robot.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'changeColor', [
                        handle], [0, 0, 0], [], bytearray(), vrep.simx_opmode_blocking)
            logger.save_masks(iteration, mask, '0')
        # mask = np.zeros((224, 224))
        # vrep.simxSetObjectIntParameter(robot.sim_client, robot.cam_handle, vrep.sim_visionintparam_entity_to_render, -1, vrep.simx_opmode_blocking)
        # for idx, handle in enumerate(robot.object_handles):
        #     vrep.simxSetObjectIntParameter(robot.sim_client, robot.cam_handle, vrep.sim_visionintparam_entity_to_render, handle, vrep.simx_opmode_blocking)
     
        #     color_img, depth_img = robot.get_camera_data()
        #     color_heightmap, depth_heightmap = utils.get_heightmap(
        #         color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        #     gray = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2GRAY)
        #     gray = gray.astype(np.uint8)
        #     blurred = cv2.medianBlur(gray, 5)
        #     thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        #     mask[thresh == 255] = idx + 1
        # logger.save_masks(iteration, mask, '0')
        # vrep.simxSetObjectIntParameter(robot.sim_client, robot.cam_handle, vrep.sim_visionintparam_entity_to_render, -1, vrep.simx_opmode_blocking)

        iteration += 1

        # If testing, read object meshes and poses from test case file
        if test_preset_cases:
            file = open(os.path.join('hard-cases', hard_cases[iteration]), 'r')
            file_content = file.readlines()
            robot.num_obj = len(file_content)
            robot.test_obj_mesh_files = []
            robot.test_obj_mesh_colors = []
            robot.test_obj_positions = []
            robot.test_obj_orientations = []
            for object_idx in range(len(file_content)):
                file_content_curr_object = file_content[object_idx].split()
                robot.test_obj_mesh_files.append(os.path.join(robot.obj_mesh_dir, file_content_curr_object[0]))
                robot.test_obj_mesh_colors.append([float(file_content_curr_object[1]), float(
                    file_content_curr_object[2]), float(file_content_curr_object[3])])
                robot.test_obj_positions.append([float(file_content_curr_object[4]), float(
                    file_content_curr_object[5]), float(file_content_curr_object[6])])
                robot.test_obj_orientations.append([float(file_content_curr_object[7]), float(
                    file_content_curr_object[8]), float(file_content_curr_object[9])])
            file.close()
            robot.obj_mesh_color = np.asarray(robot.test_obj_mesh_colors)

        robot.restart_sim()
        if test_preset_cases:
            robot.add_objects()
        else:
            robot.add_objects_mask()

        if iteration > 2000:
            break

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False, help='run in simulation?')
    parser.add_argument('--is_mask', dest='is_mask', action='store_true', default=False, help='collect for mask rcnn?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/final-mask',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store',
                        default=10, help='number of objects to add to simulation')
    parser.add_argument(
        '--tcp_host_ip',
        dest='tcp_host_ip',
        action='store',
        default='172.19.97.157',
        help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store',
                        default=30002, help='port to robot arm as TCP client (UR5)')
    parser.add_argument(
        '--rtc_host_ip',
        dest='rtc_host_ip',
        action='store',
        default='172.19.97.157',
        help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store',
                        default=30003, help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float,
                        action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,
                        help='random seed for simulation and neural net initialization')
    parser.add_argument(
        '--cpu',
        dest='force_cpu',
        action='store_true',
        default=False,
        help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument(
        '--method',
        dest='method',
        action='store',
        default='reinforcement',
        help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,
                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument(
        '--future_reward_discount',
        dest='future_reward_discount',
        type=float,
        action='store',
        default=0.5)
    parser.add_argument(
        '--experience_replay',
        dest='experience_replay',
        action='store_true',
        default=False,
        help='use prioritized experience replay?')
    parser.add_argument(
        '--heuristic_bootstrap',
        dest='heuristic_bootstrap',
        action='store_true',
        default=False,
        help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store',
                        default=30, help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true',
                        default=False, help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true',
                        default=False, help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument(
        '--save_visualizations',
        dest='save_visualizations',
        action='store_true',
        default=False,
        help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)