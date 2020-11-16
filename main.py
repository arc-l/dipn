#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
from constants import is_real, gripper_width, DEPTH_MIN, workspace_limits


def main(args):
    """
    This script is used to train DQN + GN or GN only.
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

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, test_preset_file)
    robot.reset_plane()

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(
        robot.cam_intrinsics,
        robot.cam_pose,
        robot.cam_depth_scale)  # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob = 0.5 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'primitive_action': None,
                          'best_pix_ind': None,
                          'push_success': False,
                          'grasp_success': False}

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------

    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # Determine whether grasping or pushing should be executed based on network predictions
                best_push_conf = np.max(push_predictions)
                best_grasp_conf = np.max(grasp_predictions)
                print('Primitive confidence scores: %f (push), %f (grasp)' % (best_push_conf, best_grasp_conf))
                nonlocal_variables['primitive_action'] = 'grasp'
                explore_actions = False
                if not grasp_only:
                    if is_testing and method == 'reactive':
                        if best_push_conf > 2 * best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    else:
                        if best_push_conf > best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    explore_actions = np.random.uniform() < explore_prob
                    if explore_actions:  # Exploitation (do best action) vs exploration (do other action)
                        print('Strategy: explore (exploration probability: %f)' % (explore_prob))
                        nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0, 2) == 0 else 'grasp'
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))
                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)

                # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
                # NOTE: typically not necessary and can reduce final performance.
                if heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'push' and no_change_count[0] >= 2:
                    print('Change not detected for more than two pushes. Running heuristic pushing.')
                    nonlocal_variables['best_pix_ind'] = trainer.push_heuristic(valid_depth_heightmap)
                    no_change_count[0] = 0
                    predicted_value = push_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                elif heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'grasp' and no_change_count[1] >= 2:
                    print('Change not detected for more than two grasps. Running heuristic grasping.')
                    nonlocal_variables['best_pix_ind'] = trainer.grasp_heuristic(valid_depth_heightmap)
                    no_change_count[1] = 0
                    predicted_value = grasp_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                else:
                    use_heuristic = False

                    # Get pixel location and rotation with highest affordance prediction from
                    # heuristic algorithms (rotation, y, x)
                    if nonlocal_variables['primitive_action'] == 'push':
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(
                            np.argmax(push_predictions), push_predictions.shape)
                        predicted_value = np.max(push_predictions)
                    elif nonlocal_variables['primitive_action'] == 'grasp':
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(
                            np.argmax(grasp_predictions), grasp_predictions.shape)
                        predicted_value = np.max(grasp_predictions)
                trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                logger.write_to_log('use-heuristic', trainer.use_heuristic_log)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print(
                    'Action: %s at (%d, %d, %d)' %
                    (nonlocal_variables['primitive_action'],
                     nonlocal_variables['best_pix_ind'][0],
                     nonlocal_variables['best_pix_ind'][1],
                     nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(
                    nonlocal_variables['best_pix_ind'][0] * (360.0 / trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                primitive_position = [
                    best_pix_x *
                    heightmap_resolution +
                    workspace_limits[0][0],
                    best_pix_y *
                    heightmap_resolution +
                    workspace_limits[1][0],
                    valid_depth_heightmap[best_pix_y][best_pix_x] +
                    workspace_limits[2][0]]

                # If pushing, adjust start position, and make sure z value is safe and not too low
                # or nonlocal_variables['primitive_action'] == 'place':
                if nonlocal_variables['primitive_action'] == 'push':
                    finger_width = gripper_width
                    safe_kernel_width = int(np.round((finger_width / 2) / heightmap_resolution))
                    local_region = valid_depth_heightmap[max(best_pix_y -
                                                             safe_kernel_width, 0):min(best_pix_y +
                                                                                       safe_kernel_width +
                                                                                       1, valid_depth_heightmap.shape[0]), max(best_pix_x -
                                                                                                                               safe_kernel_width, 0):min(best_pix_x +
                                                                                                                                                         safe_kernel_width +
                                                                                                                                                         1, valid_depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0]
                    else:
                        safe_z_position = np.max(local_region) + workspace_limits[2][0]
                    primitive_position[2] = safe_z_position
                    if not is_sim:
                        primitive_position[2] += 0.002
                else:
                    if not is_sim:
                        primitive_position[2] += 0.008

                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0,
                                                        nonlocal_variables['best_pix_ind'][0],
                                                        nonlocal_variables['best_pix_ind'][1],
                                                        nonlocal_variables['best_pix_ind'][2]])  # 0 - push
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1,
                                                        nonlocal_variables['best_pix_ind'][0],
                                                        nonlocal_variables['best_pix_ind'][1],
                                                        nonlocal_variables['best_pix_ind'][2]])  # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    if not grasp_only:
                        push_pred_vis = trainer.get_prediction_vis(
                            push_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                        logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                        cv2.imwrite('visualization.push.png', push_pred_vis)
                    grasp_pred_vis = trainer.get_prediction_vis(
                        grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

                # Initialize variables that influence reward
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                change_detected = False

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['push_success'] = robot.push(
                        primitive_position, best_rotation_angle, workspace_limits)
                    print('Push successful: %r' % (nonlocal_variables['push_success']))
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    nonlocal_variables['grasp_success'] = robot.grasp(
                        primitive_position, best_rotation_angle, workspace_limits)
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))

                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)
    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    # Start main training/testing loop
    while True:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim:
            robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(
            color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        valid_depth_heightmap = valid_depth_heightmap.astype(np.float32)

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > DEPTH_MIN] = 1
        print('Stuff on the table (value: %d)' % (np.sum(stuff_count)))
        empty_threshold = 300
        if is_sim and is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 10):
            no_change_count = [0, 0]
            if is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
                if is_testing:  # If at end of test run, re-load original weights (before test run)
                    trainer.loss_list = []
                    trainer.optimizer.zero_grad()
                    trainer.model.load_state_dict(torch.load(snapshot_file))
                    robot.reset_plane()
                first_step = True
            else:
                # print('Not enough stuff on the table (value: %d)! Pausing for 30 seconds.' % (np.sum(stuff_count)))
                # time.sleep(30)
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (np.sum(stuff_count)))
                input("Please maually reset scene")
                # robot.restart_real()

            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True  # Exit after training thread (backprop and saving labels)

            if 'prev_color_img' in locals():
                # Detect changes
                depth_diff = abs(depth_heightmap - prev_depth_heightmap)
                depth_diff[np.isnan(depth_diff)] = 0
                depth_diff[depth_diff > 0.3] = 0
                depth_diff[depth_diff < 0.01] = 0
                depth_diff[depth_diff > 0] = 1
                change_threshold = 300
                change_value = np.sum(depth_diff)
                if prev_primitive_action == 'push':
                    change_detected = change_value > change_threshold
                elif prev_primitive_action == 'grasp':
                    change_detected = prev_grasp_success
                print('Change detected: %r (value: %d)' % (change_detected, change_value))

                if change_detected:
                    if prev_primitive_action == 'push':
                        no_change_count[0] = 0
                    elif prev_primitive_action == 'grasp':
                        no_change_count[1] = 0
                else:
                    if prev_primitive_action == 'push':
                        no_change_count[0] += 1
                    elif prev_primitive_action == 'grasp':
                        no_change_count[1] += 1

                # Compute training labels
                label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_push_success, prev_grasp_success, change_detected,
                                                                         prev_push_predictions, prev_grasp_predictions, color_heightmap, valid_depth_heightmap, prev_valid_depth_heightmap, use_push=(not grasp_only))
                trainer.label_value_log.append([label_value])
                logger.write_to_log('label-value', trainer.label_value_log)
                trainer.reward_value_log.append([prev_reward_value])
                logger.write_to_log('reward-value', trainer.reward_value_log)

                trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value, use_push=(not grasp_only))                

                del prev_color_img
                nonlocal_variables['push_success'] = False
                nonlocal_variables['grasp_success'] = False
                nonlocal_variables['primitive_action'] = None
                nonlocal_variables['best_pix_ind'] = None
            continue

        if not exit_called:

            # Run forward pass with network to get affordances
            push_predictions, grasp_predictions = trainer.forward(color_heightmap, valid_depth_heightmap, is_volatile=True, use_push=(not grasp_only))

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():

            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 300
            change_value = np.sum(depth_diff)
            if prev_primitive_action == 'push':
                change_detected = change_value > change_threshold
            elif prev_primitive_action == 'grasp':
                change_detected = prev_grasp_success
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if change_detected:
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'push':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1

            # Compute training labels
            label_value, prev_reward_value = trainer.get_label_value(prev_primitive_action, prev_push_success, prev_grasp_success, change_detected,
                                                                     prev_push_predictions, prev_grasp_predictions, color_heightmap, valid_depth_heightmap, prev_valid_depth_heightmap, use_push=(not grasp_only))
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)

            # Backpropagate
            trainer.backprop(
                prev_color_heightmap,
                prev_valid_depth_heightmap,
                prev_primitive_action,
                prev_best_pix_ind,
                label_value, use_push=(not grasp_only))

            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9998, trainer.iteration), 0.1) if explore_rate_decay else 0.5

            # Do sampling for experience replay
            if experience_replay and not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                    if method == 'reactive':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1  # random.randint(1, 2) # 2
                    elif method == 'reinforcement':
                        sample_reward_value = 0 if prev_reward_value > 0 else 0.1
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1
                    if method == 'reactive':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1
                    elif method == 'reinforcement':
                        sample_reward_value = 0 if prev_reward_value == 1 else 1

                # Get samples of the same primitive but with different results
                if sample_primitive_action == 'push' and sample_reward_value == 0.1:
                    # sample_ind = np.argwhere(np.asarray(trainer.executed_action_log)[:trainer.iteration - 1, 0] == sample_primitive_action_id)
                    sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[:trainer.iteration - 1, 0] > sample_reward_value,
                                                            np.asarray(trainer.executed_action_log)[:trainer.iteration - 1, 0] == sample_primitive_action_id))
                else:
                    sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[:trainer.iteration - 1, 0] == sample_reward_value,
                                                            np.asarray(trainer.executed_action_log)[:trainer.iteration - 1, 0] == sample_primitive_action_id))
                # don't care the reward
                # sample_ind = np.argwhere(np.asarray(trainer.executed_action_log)[:trainer.iteration - 1, 0] == sample_primitive_action_id)

                if sample_ind.size > 0:

                    # Find sample with highest surprise value
                    if method == 'reactive':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[
                                                        sample_ind[:, 0]] - (1 - sample_reward_value))
                    elif method == 'reinforcement':
                        sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[
                                                        sample_ind[:, 0]] - np.asarray(trainer.label_value_log)[sample_ind[:, 0]])
                    sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
                    sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
                    pow_law_exp = 2
                    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
                    sample_iteration = sorted_sample_ind[rand_sample_ind]
                    print('Experience replay: iteration %d (surprise value: %f)' %
                          (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

                    # Load sample RGB-D heightmap
                    sample_color_heightmap = cv2.imread(
                        os.path.join(
                            logger.color_heightmaps_directory,
                            '%06d.0.color.png' %
                            (sample_iteration)))
                    sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    sample_depth_heightmap = cv2.imread(
                        os.path.join(
                            logger.depth_heightmaps_directory, '%06d.0.depth.png' %
                            (sample_iteration)), -1)
                    sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000

                    # Compute forward pass with sample
                    with torch.no_grad():
                        sample_push_predictions, sample_grasp_predictions = trainer.forward(
                            sample_color_heightmap, sample_depth_heightmap, is_volatile=True, use_push=(not grasp_only))

                    # Load next sample RGB-D heightmap
                    next_sample_color_heightmap = cv2.imread(
                        os.path.join(
                            logger.color_heightmaps_directory, '%06d.0.color.png' %
                            (sample_iteration + 1)))
                    next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
                    next_sample_depth_heightmap = cv2.imread(
                        os.path.join(
                            logger.depth_heightmaps_directory, '%06d.0.depth.png' %
                            (sample_iteration + 1)), -1)
                    next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32) / 100000

                    sample_reward_value = np.asarray(trainer.reward_value_log)[sample_iteration, 0]
                    sample_push_success = sample_reward_value > 0
                    sample_grasp_success = sample_reward_value == 1
                    sample_change_detected = sample_push_success
                    new_sample_label_value, new_sample_reward_value = trainer.get_label_value(
                        sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected, sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap, sample_depth_heightmap, use_push=(not grasp_only))

                    # Get labels for sample and backpropagate
                    sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
                    # trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration])
                    trainer.backprop(
                        sample_color_heightmap,
                        sample_depth_heightmap,
                        sample_primitive_action,
                        sample_best_pix_ind,
                        new_sample_label_value, use_push=(not grasp_only))

                    # Recompute prediction value and label for replay buffer
                    if sample_primitive_action == 'push':
                        print("Surprise value from %f to %f" %
                              (abs(trainer.predicted_value_log[sample_iteration][0] -
                                   trainer.label_value_log[sample_iteration][0]), abs(np.max(sample_push_predictions -
                                                                                             trainer.label_value_log[sample_iteration][0]))))
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
                        trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                        trainer.reward_value_log[sample_iteration] = [new_sample_reward_value]
                        logger.write_to_log('predicted-value', trainer.predicted_value_log)
                        logger.write_to_log('reward-value', trainer.reward_value_log)
                        logger.write_to_log('label-value', trainer.label_value_log)
                    elif sample_primitive_action == 'grasp':
                        print("Surprise value from %f to %f" %
                              (abs(trainer.predicted_value_log[sample_iteration][0] -
                                   trainer.label_value_log[sample_iteration][0]), abs(np.max(sample_grasp_predictions -
                                                                                             trainer.label_value_log[sample_iteration][0]))))
                        trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
                        trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                        trainer.reward_value_log[sample_iteration] = [new_sample_reward_value]
                        logger.write_to_log('predicted-value', trainer.predicted_value_log)
                        logger.write_to_log('reward-value', trainer.reward_value_log)
                        logger.write_to_log('label-value', trainer.label_value_log)
                    print(
                        "Replay update: %f, %f, %f" %
                        (trainer.predicted_value_log[sample_iteration][0],
                         trainer.label_value_log[sample_iteration][0],
                         trainer.reward_value_log[sample_iteration][0]))

                else:
                    print('Not enough prior training samples. Skipping experience replay.')

            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.iteration, trainer.model, method)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        if grasp_only:
            prev_push_predictions = 0
        else:
            prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))

        logger.write_to_log('batch-loss', trainer.loss_log)
        if trainer.iteration > 1501:
            exit_called = True


def post_train(args):
    """
    For grasp-only training, this offline-training can be used train the network as supervised learning. But, we didn't use it.
    """
    # TODO only work for sim now
    # --------------- Setup options ---------------
    is_sim = args.is_sim  # Run in simulation?
    # Directory containing 3D mesh files (.obj) of objects to be added to simulation
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

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    for i in range(100000):
        # sample_primitive_action = np.random.choice(['push', 'grasp'], p=[0.3, 0.7])
        sample_primitive_action = 'grasp'
        if sample_primitive_action == 'push':
            sample_primitive_action_id = 0
            print('Push!!!')
        elif sample_primitive_action == 'grasp':
            sample_primitive_action_id = 1
            print('Grasp!!!')

        # sample_label = np.random.choice([0, 1], p=[0.8, 0.2])
        # sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.reward_value_log)[:trainer.iteration - 1, 0] == sample_label,
        #                                                     np.asarray(trainer.executed_action_log)[:trainer.iteration - 1, 0] == sample_primitive_action_id))
        sample_ind = np.argwhere(np.asarray(trainer.executed_action_log)[:trainer.iteration - 1, 0] == sample_primitive_action_id)

        if sample_ind.size > 0:

            # Find sample with highest surprise value
            if method == 'reactive':
                sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[
                                                sample_ind[:, 0]] - (1 - sample_reward_value))
            elif method == 'reinforcement':
                sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[
                                                sample_ind[:, 0]] - np.asarray(trainer.label_value_log)[sample_ind[:, 0]])
            sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
            sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
            pow_law_exp = 2
            rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
            sample_iteration = sorted_sample_ind[rand_sample_ind]
            print('Experience replay: iteration %d (surprise value: %f)' %
                  (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
            print(np.sum(sample_surprise_values > 0.5))
            print(np.sum(sample_surprise_values))

            # Load sample RGB-D heightmap
            sample_color_heightmap = cv2.imread(
                os.path.join(
                    logger.color_heightmaps_directory,
                    '%06d.0.color.png' %
                    (sample_iteration)))
            sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
            sample_depth_heightmap = cv2.imread(
                os.path.join(
                    logger.depth_heightmaps_directory, '%06d.0.depth.png' %
                    (sample_iteration)), -1)
            sample_depth_heightmap = sample_depth_heightmap.astype(np.float32) / 100000

            # Compute forward pass with sample
            with torch.no_grad():
                sample_push_predictions, sample_grasp_predictions = trainer.forward(
                    sample_color_heightmap, sample_depth_heightmap, is_volatile=True, use_push=(not grasp_only))

            # Load next sample RGB-D heightmap
            next_sample_color_heightmap = cv2.imread(
                os.path.join(
                    logger.color_heightmaps_directory, '%06d.0.color.png' %
                    (sample_iteration + 1)))
            next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
            next_sample_depth_heightmap = cv2.imread(
                os.path.join(
                    logger.depth_heightmaps_directory, '%06d.0.depth.png' %
                    (sample_iteration + 1)), -1)
            next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32) / 100000

            sample_reward_value = np.asarray(trainer.reward_value_log)[sample_iteration, 0]
            sample_push_success = sample_reward_value > 0
            sample_grasp_success = sample_reward_value == 1
            sample_change_detected = sample_push_success
            new_sample_label_value, new_sample_reward_value = trainer.get_label_value(
                sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected, sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap, sample_depth_heightmap, use_push=(not grasp_only))

            # Get labels for sample and backpropagate
            sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
            # trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind, trainer.label_value_log[sample_iteration])
            trainer.backprop(
                sample_color_heightmap,
                sample_depth_heightmap,
                sample_primitive_action,
                sample_best_pix_ind,
                new_sample_label_value, use_push=(not grasp_only))

            # Recompute prediction value and label for replay buffer
            if sample_primitive_action == 'push':
                print("Surprise value from %f to %f" %
                      (abs(trainer.predicted_value_log[sample_iteration][0] -
                           trainer.label_value_log[sample_iteration][0]), abs(np.max(sample_push_predictions -
                                                                                     trainer.label_value_log[sample_iteration][0]))))
                trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
                trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                trainer.reward_value_log[sample_iteration] = [new_sample_reward_value]
            elif sample_primitive_action == 'grasp':
                print("Surprise value from %f to %f" %
                      (abs(trainer.predicted_value_log[sample_iteration][0] -
                           trainer.label_value_log[sample_iteration][0]), abs(np.max(sample_grasp_predictions -
                                                                                     trainer.label_value_log[sample_iteration][0]))))
                trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
                trainer.label_value_log[sample_iteration] = [new_sample_label_value]
                trainer.reward_value_log[sample_iteration] = [new_sample_reward_value]
            print(
                "Replay update: %f, %f, %f" %
                (trainer.predicted_value_log[sample_iteration][0],
                 trainer.label_value_log[sample_iteration][0],
                 trainer.reward_value_log[sample_iteration][0]))

        else:
            print('Not enough prior training samples. Skipping experience replay.')

        if i % 500 == 0:
            logger.save_model(i + 500, trainer.model, method)
            print("Saved at iteration %f" % (i + 10000))
            if trainer.use_cuda:
                trainer.model = trainer.model.cuda()

        if i % 50 == 0:
            print('===================================')
            print("Post iteration %f" % (i + 10000))
            logger.write_to_log('predicted-value', trainer.predicted_value_log)
            logger.write_to_log('reward-value', trainer.reward_value_log)
            logger.write_to_log('label-value', trainer.label_value_log)


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False, help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/object',
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
        default=0.2)
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
    # post_train (args)
