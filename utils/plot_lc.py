#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


''' Some global variables '''
FIGURE_SIZE = (3.45, 1.35)      # Figure width and height. 
                                # This is a good value for 2-column paper.
FONT_SIZE = 8                   # Size of text
LEGEND_FONT_SIZE = 7            # We might need different font sizes for different text
OUTDIR = "/home/mluser/VPG/"

''' 
Pyplot parameters for legend to better utilize space.
'''
plt.rcParams["legend.labelspacing"] = 0.2
plt.rcParams["legend.handlelength"] = 1.75
plt.rcParams["legend.handletextpad"] = 0.5
plt.rcParams["legend.columnspacing"] = 0.75
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


# Plot options (change me)
interval_size = 200 # Report performance over the last 200 training steps
max_plot_iteration = 2500 # Maximum number of training steps to report performance

# Parse session directories
parser = argparse.ArgumentParser(description='Plot performance of a session over training time.')
parser.add_argument('session_directories', metavar='N', type=str, nargs='+', help='path to session directories for which to plot performance')
args = parser.parse_args()
session_directories = args.session_directories

# Define plot colors (Tableau palette)
colors = [[078.0/255.0,121.0/255.0,167.0/255.0], # blue
          [255.0/255.0,087.0/255.0,089.0/255.0], # red
          [089.0/255.0,169.0/255.0,079.0/255.0], # green
          [237.0/255.0,201.0/255.0,072.0/255.0], # yellow
          [242.0/255.0,142.0/255.0,043.0/255.0], # orange
          [176.0/255.0,122.0/255.0,161.0/255.0], # purple
          [255.0/255.0,157.0/255.0,167.0/255.0], # pink 
          [118.0/255.0,183.0/255.0,178.0/255.0], # cyan
          [156.0/255.0,117.0/255.0,095.0/255.0], # brown
          [186.0/255.0,176.0/255.0,172.0/255.0]] # gray
colors = ["C2", "C0", "C1"]

# Determine whether each session directory is trained in 'reactive' or 'reinforcement' mode (reward schemes differ between methods)
methods = []
for session_directory in session_directories:

    # Check name of saved weights
    model_list = os.listdir(os.path.join(session_directory, 'models'))
    if len(model_list) > 0:
        if 'reactive' in model_list[0]:
            methods.append('reactive')
        elif 'reinforcement' in model_list[0]:
            methods.append('reinforcement')
        else:
            print('Error: cannot determine whether session was trained in \'reactive\' or \'reinforcement\' mode.')
    else:
        print('Error: no model weights saved, cannot determine whether session was trained in \'reactive\' or \'reinforcement\' mode.')

# Create plot design
fig = plt.figure(figsize = (FIGURE_SIZE[0], FIGURE_SIZE[1]))
ax = fig.add_subplot(111)
ax.set_ylim((0, 1))
# ax.set_ylabel('Grasp success rate (%)', fontsize = FONT_SIZE)
ax.set_xlim((0, max_plot_iteration))
ax.set_xlabel('Number of training actions (grasp and push)', fontsize = FONT_SIZE)
ax.yaxis.grid(True, alpha = 0.8, zorder=-100)
ax.xaxis.grid(True, alpha = 0.8, zorder=-100)
# ax.tick_params(labelsize = FONT_SIZE)
ax.set_yticks([i * 0.01 for i in range(0, 101, 20)])
ax.set_yticklabels([i for i in range(0, 101, 20)])
ax.xaxis.set_label_coords(0.5, -0.22)
ax.yaxis.set_label_coords(-0.08, 0.4)
ax.tick_params(labelsize = LEGEND_FONT_SIZE, pad=1)
ax.set_ylabel('Grasp success %', fontsize = FONT_SIZE)
# ax = plt.gca()
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_color('#000000')
# plt.rcParams.update({'font.size': 18})
# plt.rcParams['mathtext.default']='regular'
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
legend = []

for session_idx in range(len(session_directories)):
    session_directory = session_directories[session_idx]
    method = methods[session_idx]
    color = colors[session_idx % 10]

    # Get logged data
    transitions_directory = os.path.join(session_directory, 'transitions')
    executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
    max_iteration = min(executed_action_log.shape[0] - 2, max_plot_iteration)
    executed_action_log = executed_action_log[0:max_iteration,:]
    reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
    reward_value_log = reward_value_log[0:max_iteration]

    # Initialize plot variables
    grasp_to_push_ratio = np.zeros((max_iteration))
    grasp_success = np.zeros((max_iteration))
    push_then_grasp_success = np.zeros((max_iteration))
    q_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
    q_value_log = q_value_log[0:max_iteration]

    for step in range(max_iteration):

        # Get indicies for previous x grasps, where x is the interval size
        grasp_attempt_ind = np.argwhere(executed_action_log[:,0] == 1)
        grasp_attempt_ind_before_step = grasp_attempt_ind[np.argwhere(grasp_attempt_ind[:,0] < step)]
        grasp_attempt_ind_over_interval = grasp_attempt_ind_before_step[max(0,len(grasp_attempt_ind_before_step)-interval_size):len(grasp_attempt_ind_before_step),0]

        # Count number of times grasp attempts were successful
        # NOTE: reward_value_log just stores some value which is indicative of successful grasping, which could be a class ID (reactive) or actual reward value (from MDP, reinforcement)
        if method == 'reactive':
            grasp_success_over_interval = np.sum(reward_value_log[grasp_attempt_ind_over_interval] == 0)/float(min(interval_size,max(step,1))) # Class ID for successful grasping is 0 (reactive policy)
        elif method == 'reinforcement':
            grasp_success_over_interval = np.sum(reward_value_log[grasp_attempt_ind_over_interval] >= 0.5)/float(min(interval_size,max(step,1))) # Reward value for successful grasping is anything larger than 0.5 (reinforcement policy)
        if step < interval_size:
            grasp_success_over_interval *= (float(step)/float(interval_size))
        grasp_success[step] = grasp_success_over_interval

        # Get grasp to push ratio over previous x attempts, where x is the interval size
        grasp_to_push_ratio_over_interval = float(np.sum(executed_action_log[max(0,step-interval_size):step,0] == 1))/float(min(interval_size,max(step,1)))
        grasp_to_push_ratio[step] = grasp_to_push_ratio_over_interval

        # Get indicies for push-then-grasp cases
        push_attempt_ind = np.argwhere(executed_action_log[0:(max_iteration-1),0] == 0)
        grasp_after_push_attempt_ind = push_attempt_ind[np.argwhere(executed_action_log[push_attempt_ind[:,0] + 1,0] == 1),:] + 1
        grasp_after_push_attempt_ind_before_step = grasp_after_push_attempt_ind[np.argwhere(grasp_after_push_attempt_ind[:,0] < step)]
        grasp_after_push_attempt_ind_over_interval = grasp_after_push_attempt_ind_before_step[max(0,len(grasp_after_push_attempt_ind_before_step)-interval_size):len(grasp_after_push_attempt_ind_before_step),0]

        # Count number of times grasp after push attempts were successful
        if method == 'reactive':
            grasp_after_push_success_over_interval = np.sum(reward_value_log[grasp_after_push_attempt_ind_over_interval] == 0)/float(min(interval_size,max(step,1)))
        elif method == 'reinforcement':
            grasp_after_push_success_over_interval = np.sum(reward_value_log[grasp_after_push_attempt_ind_over_interval] >= 0.5)/float(min(interval_size,max(step,1)))
        if step < interval_size:
            grasp_after_push_success_over_interval *= (float(step)/float(interval_size))
        push_then_grasp_success[step] = grasp_after_push_success_over_interval

        if step > 200:
            idx = np.argwhere(reward_value_log[step-100:step] >= 0.5)
            idx = np.squeeze(idx)
            print(np.min(q_value_log[idx]), np.mean(q_value_log[idx]))


    # Plot grasp information
    ax.plot(range(0, max_iteration), grasp_success, color=color, linewidth=1) # color='blue', linewidth=3)
    # plt.fill_between(range(0, max_iteration), push_then_grasp_success, 0, color=color, alpha=0.5)
    # plt.plot(range(0, max_iteration), push_then_grasp_success, dashes=[8,7], color=color, linewidth=3, alpha=1, dash_capstyle='round', dash_joinstyle='round',label='_nolegend_') # color='blue', dashes=[5,5], linewidth=2, dash_capstyle='butt')
    ax.plot(range(0, max_iteration), push_then_grasp_success, linestyle="dashed", color=color, linewidth=1, alpha=1,label='_nolegend_')
    # plt.plot(range(0, max_iteration), grasp_to_push_ratio, dashes=[1,5], color=color, linewidth=2, alpha=0.5, dash_capstyle='round', dash_joinstyle='round') # color='blue', dashes=[5,5], linewidth=2, dash_capstyle='butt')
    legend.append(session_directories[session_idx])

legend = ['GN', 'VPG', 'DQN+GN']
# ax.legend(legend, loc='lower right', fontsize=LEGEND_FONT_SIZE)
ax.legend(legend, fontsize=LEGEND_FONT_SIZE)
fig.savefig(os.path.join(OUTDIR, "real-learning-curve.pdf"), bbox_inches="tight", pad_inches=0.05)
# plt.show()

# python plot_lc.py '../VPG/logs/vpg+&pp/new' 'logs/vpg' '../VPG/logs/vpg+/updated-vpg' 