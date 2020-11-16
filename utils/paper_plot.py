# Import deepcopy for copying data structures
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import random       # Just for demo purpose

'''
Style elements
    Here we can setup a drawing style for each algorithm.
    This helps to make the drawing style of an algorithm consistent.
'''
defaultStyle = {
    'label' : 'default',    # The name of the algorithm
    'ls' : '-',             # Line style, '-' means a solid line
    'linewidth' : 1,        # Line width
    'color' : 'k',          # Line color, 'k' means color
    'zorder' : 100,         # The 'height' of the plot. 
                            # Affects whether items are in the front of / behind each other.
    # You can add more style items here, e.g., markers.
}
# Here, we setup the style for an algorithm. Let's call it Alg.1.
alg1Style = deepcopy(defaultStyle)          # First copy all default styles.
alg1Style['label'] = r'\textsc{Optimal}'    # Setup algorithm name. 
                                            # We use \textsc here which is a latex command. 
                                            # This is fine since we use latex to generate text.
alg1Style['color'] = 'tab:orange'           # Customized line color
                                            # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
# Another algorithm
alg2Style = deepcopy(defaultStyle)
alg2Style['label'] = r'\textsc{Greedy}'
alg2Style['color'] = 'tab:green'

''' Some global variables '''
FIGURE_SIZE = (3.45, 1.15)      # Figure width and height. 
                                # This is a good value for 2-column paper.
FONT_SIZE = 8                   # Size of text
LEGEND_FONT_SIZE = 7            # We might need different font sizes for different text
OUTDIR = "figures/"

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

'''
Draw push prediction
'''
fig = plt.figure(figsize = (FIGURE_SIZE[0], FIGURE_SIZE[1]))
ax = fig.add_subplot(111)
folder_5cm = "/home/mluser/VPG/logs_push/random/data/5cm/"
for folder_dir, sub_dir, files in os.walk(folder_5cm):
    x = list()
    avg = list()
    std_dev = list()
    for file_name in files:
        if file_name[-3:] == "txt":
            file_dir = os.path.join(folder_dir, file_name)
            with open(file_dir, "r") as in_file:
                lines = in_file.readlines()
                data = np.array([float(line) for line in lines])
            if file_name[:2] == "no":
                ax.plot([0, 2000], [np.average(data), np.average(data)], label="5cm static", color="C0", linestyle="dotted", linewidth=0.75)
            elif file_name[:4] == "move":
                ax.plot([0, 2000], [np.average(data), np.average(data)], label="5cm trans", color="C0", linestyle="dashed", linewidth=0.75)
            else:
                x.append(int(file_name.split(".")[0]) * 100)
                avg.append(np.average(data))
                std_dev.append(np.std(data))
    x, avg, std_dev = (np.array(t) for t in zip(*sorted(zip(x, avg, std_dev))))
    ax.plot(x, avg, label="5cm DIPN", color="C0", linewidth=0.75)
    ax.fill_between(x, avg-std_dev, avg+std_dev, color="C0", alpha=0.3, linewidth=0)
folder_10cm = "/home/mluser/VPG/logs_push/random/data/10cm/"
for folder_dir, sub_dir, files in os.walk(folder_10cm):
    x = list()
    avg = list()
    std_dev = list()
    for file_name in files:
        if file_name[-3:] == "txt":
            file_dir = os.path.join(folder_dir, file_name)
            with open(file_dir, "r") as in_file:
                lines = in_file.readlines()
                data = np.array([float(line) for line in lines])
            if file_name[:2] == "no":
                ax.plot([0, 2000], [np.average(data), np.average(data)], label="10cm static", color="C1", linestyle="dotted", linewidth=0.75)
            elif file_name[:4] == "move":
                ax.plot([0, 2000], [np.average(data), np.average(data)], label="10cm trans", color="C1", linestyle="dashed", linewidth=0.75)
            else:
                x.append(int(file_name.split(".")[0]) * 100)
                avg.append(np.average(data))
                std_dev.append(np.std(data))
    x, avg, std_dev = (np.array(t) for t in zip(*sorted(zip(x, avg, std_dev))))
    ax.plot(x, avg, label="10cm DIPN", color="C1", linewidth=0.75)
    ax.fill_between(x, avg-std_dev, avg+std_dev, color="C1", alpha=0.3, linewidth=0)
ax.set_xlabel("Number of push action training samples", fontsize = FONT_SIZE-2)
ax.set_ylabel("Prediction error", fontsize = FONT_SIZE-2) 
ax.xaxis.set_label_coords(0.5, -0.23)
ax.yaxis.set_label_coords(-0.12, 0.4)

ax.tick_params(labelsize = LEGEND_FONT_SIZE-1.5, pad=1)
handles, labels = ax.get_legend_handles_labels()
order = [2, 5, 1, 4, 0, 3]
ax.set_xlim(100, 1500)
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
# set_size(2, 1, ax)
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize = LEGEND_FONT_SIZE - 1.5, ncol = 1, loc='center right', bbox_to_anchor=(1.5, 0.5))
# ax.set_yscale("log")
# ax.yaxis.grid(True, alpha = 0.8)
# Directly save the figure to a file.
# fig.savefig(os.path.join(OUTDIR, "push-prediction-alone.pdf"), bbox_inches="tight", pad_inches=0.05)
ax.set_xscale("log")
ax.set_xticks([100, 500, 1000, 1500])
from matplotlib.ticker import ScalarFormatter
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())
ax.set_yticks([0.2 * i for i in range(1, 5)])
ax.set_yticklabels([round(0.1 * i, 1) for i in range(1, 5)])
ax.set_ylim(0.15, 0.9)
ax.yaxis.grid(True, alpha = 0.8, zorder=-100, linewidth=0.5)
# ax.yaxis.grid(True, alpha = 0.8)
# Directly save the figure to a file.
fig.savefig(os.path.join(OUTDIR, "push-prediction-alone-log.pdf"), bbox_inches="tight", pad_inches=0.05)
# plt.show()
fig.set_size_inches(FIGURE_SIZE[0] + 2, FIGURE_SIZE[1])
order = [2, 1, 0, 5, 4, 3]
ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize = LEGEND_FONT_SIZE, ncol = 1, loc='center right', bbox_to_anchor=(1.5, 0.5))
fig.savefig(os.path.join(OUTDIR, "push-prediction-alone-wide.pdf"), bbox_inches="tight", pad_inches=0.05)
plt.cla()


# '''
# Draw learning curves
# '''
# pass



# '''
# Draw random case bar charts
# '''
# fig = plt.figure(figsize = FIGURE_SIZE)
# ax = fig.add_subplot(111)
# x = np.array([10, 20, 30])
# bar_width = 2
# # Original VPG
# data = [36.4, 69.0, 66.3]
# ax.bar(x - bar_width / 2 * 3, data, bar_width, edgecolor = 'black', color="C0", label="VPG"    , zorder=100)
# for i, d in enumerate(data):
#     ax.text(x[i] - bar_width / 2 * 3, d + 2, str(d), horizontalalignment="center", fontsize=LEGEND_FONT_SIZE - 1)
# # Original VPG + push prediction
# data = [72.7, 82.8, 76.1]
# ax.bar(x - bar_width / 2    , data, bar_width, edgecolor = 'black', color="C1", label="VPG+PP" , zorder=100)
# for i, d in enumerate(data):
#     ax.text(x[i] - bar_width / 2    , d + 2, str(d), horizontalalignment="center", fontsize=LEGEND_FONT_SIZE - 1)
# # Updated VPG
# data = [0, 0, 0]
# ax.bar(x + bar_width / 2    , data, bar_width, edgecolor = 'black', color="C2", label="VPG*"   , zorder=100)
# for i, d in enumerate(data):
#     ax.text(x[i] + bar_width / 2    , d + 2, str(d), horizontalalignment="center", fontsize=LEGEND_FONT_SIZE - 1)
# # Updated VPG + push prediction
# data = [0, 0, 0]
# ax.bar(x + bar_width / 2 * 3, data, bar_width, edgecolor = 'black', color="C3", label="VPG*+PP", zorder=100)
# for i, d in enumerate(data):
#     ax.text(x[i] + bar_width / 2 * 3, d + 2, str(d), horizontalalignment="center", fontsize=LEGEND_FONT_SIZE - 1)
# ax.set_xticks(x)
# xlabels = [item.get_text() for item in ax.get_xticklabels()]
# xlabels[0] = "Completion"
# xlabels[1] = "Grasp success"
# xlabels[2] = "Action efficiency"
# ax.set_xticklabels(xlabels)
# ax.set_ylabel("Metric value (%)", fontsize = FONT_SIZE)
# ax.set_ylim([0, 125])
# ax.set_yticks([i for i in range(0, 101, 20)])
# ax.tick_params(labelsize = FONT_SIZE)
# ax.legend(fontsize = LEGEND_FONT_SIZE, ncol = 4)
# ax.yaxis.grid(True, alpha = 0.8, zorder=-100)
# # Directly save the figure to a file.
# fig.savefig(os.path.join(OUTDIR, "random-case-barchart.pdf"), bbox_inches="tight", pad_inches=0.05)
# # plt.show()
# plt.cla()


# '''
# Draw hard case bar charts
# '''
# fig = plt.figure(figsize = FIGURE_SIZE)
# ax = fig.add_subplot(111)
# x = np.array([10, 20, 30])
# bar_width = 2
# # Original VPG
# data = [78.5, 65.7, 59.4]
# ax.bar(x - bar_width / 2 * 3, data, bar_width, edgecolor = 'black', color="C0", label="VPG"    , zorder=100)
# for i, d in enumerate(data):
#     ax.text(x[i] - bar_width / 2 * 3, d + 2, str(d), horizontalalignment="center", fontsize=LEGEND_FONT_SIZE - 1)
# # Original VPG + push prediction
# data = [90.9, 80.5, 65.2]
# ax.bar(x - bar_width / 2    , data, bar_width, edgecolor = 'black', color="C1", label="VPG+PP" , zorder=100)
# for i, d in enumerate(data):
#     ax.text(x[i] - bar_width / 2    , d + 2, str(d), horizontalalignment="center", fontsize=LEGEND_FONT_SIZE - 1)
# # Updated VPG
# data = [0, 0, 0]
# ax.bar(x + bar_width / 2    , data, bar_width, edgecolor = 'black', color="C2", label="VPG*"   , zorder=100)
# for i, d in enumerate(data):
#     ax.text(x[i] + bar_width / 2    , d + 2, str(d), horizontalalignment="center", fontsize=LEGEND_FONT_SIZE - 1)
# # Updated VPG + push prediction
# data = [0, 0, 0]
# ax.bar(x + bar_width / 2 * 3, data, bar_width, edgecolor = 'black', color="C3", label="VPG*+PP", zorder=100)
# for i, d in enumerate(data):
#     ax.text(x[i] + bar_width / 2 * 3, d + 2, str(d), horizontalalignment="center", fontsize=LEGEND_FONT_SIZE - 1)
# ax.set_xticks(x)
# xlabels = [item.get_text() for item in ax.get_xticklabels()]
# xlabels[0] = "Completion"
# xlabels[1] = "Grasp success"
# xlabels[2] = "Action efficiency"
# ax.set_xticklabels(xlabels)
# ax.set_ylabel("Metric value (%)", fontsize = FONT_SIZE)
# ax.set_ylim([0, 125])
# ax.set_yticks([i for i in range(0, 101, 20)])
# ax.tick_params(labelsize = FONT_SIZE)
# ax.legend(fontsize = LEGEND_FONT_SIZE, ncol = 4)
# ax.yaxis.grid(True, alpha = 0.8, zorder=-100)
# # Directly save the figure to a file.
# fig.savefig(os.path.join(OUTDIR, "hard-case-barchart.pdf"), bbox_inches="tight", pad_inches=0.05)
# # plt.show()
# plt.cla()



# ''' The real drawing part starts here. '''
# # Put your data over here.
# x = [i for i in range(10, 100, 10)]
# alg1ComputationTime = [2**i for i in range(1, 10)]
# alg2ComputationTime = [10 * i for i in range(1, 10)]
# alg1StdDev = [random.random() * i * 2 for i in range(1, 10)]
# alg2StdDev = [random.random() * i * 2 for i in range(1, 10)]
# # Start to create the figure
# fig = plt.figure(figsize = FIGURE_SIZE)
# ax = fig.add_subplot(111)
# ax.plot(x, alg1ComputationTime, **alg1Style)
# ax.plot(x, alg2ComputationTime, **alg2Style)
# ax.errorbar(x, alg1ComputationTime, yerr = alg1StdDev, color = alg1Style['color'], capsize = 2, ls = 'none', markeredgewidth = 1, elinewidth = 1)
# ax.errorbar(x, alg2ComputationTime, yerr = alg2StdDev, color = alg2Style['color'], capsize = 2, ls = 'none', markeredgewidth = 1, elinewidth = 1)
# # Set x and y label. We use latex to generate text
# ax.set_xlabel(r"Number of Robots $(n)$", fontsize = FONT_SIZE)
# ax.set_ylabel("Computation Time (s)", fontsize = FONT_SIZE)
# ax.tick_params(labelsize = FONT_SIZE)
# ax.legend(fontsize = LEGEND_FONT_SIZE, ncol = 2)
# ax.set_yscale("log")
# ax.yaxis.grid(True, alpha = 0.8)
# # Directly save the figure to a file.
# fig.savefig("result-computation-time.pdf", bbox_inches="tight", pad_inches=0.05)
# plt.cla()

# ''' Another bar chart. '''
# # Put your data over here.
# x = np.array([i for i in range(10, 100, 10)])
# alg1OptimalityRatio = [1 for i in range(1, 10)]
# alg2OptimalityRatio = [0.1 * i + 1 for i in range(1, 10)]
# # Start to create the figure
# fig = plt.figure(figsize = FIGURE_SIZE)
# ax = fig.add_subplot(111)
# bar_width = 3
# ax.bar(x - bar_width / 2, alg1OptimalityRatio, bar_width, edgecolor = 'black', **alg1Style)
# ax.bar(x + bar_width / 2, alg2OptimalityRatio, bar_width, edgecolor = 'black', **alg2Style)
# ax.set_xticks(x)
# ax.set_xlabel(r"Number of Robots $(n)$", fontsize = FONT_SIZE)
# ax.set_ylabel("Optimality Ratio", fontsize = FONT_SIZE)
# ax.tick_params(labelsize = FONT_SIZE)
# ax.legend(fontsize = LEGEND_FONT_SIZE, ncol = 2)
# ax.yaxis.grid(True, alpha = 0.8)
# # Directly save the figure to a file.
# fig.savefig("result-optimality-ratio.pdf", bbox_inches="tight", pad_inches=0.05)
# plt.cla()