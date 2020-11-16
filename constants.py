import numpy as np

is_real = False
distance = 5

if is_real:
    # robot specs
    gripper_width = 0.02  # cm
    gripper_thickness = 0.01  # cm
    gripper_distance = 0.1  # cm

    # depth filter, count valid object
    DEPTH_MIN = 0.01

    # workspace
    workspace_limits = np.asarray([[-0.235, 0.213], [-0.678, -0.230], [0.18, 0.4]])
else:
    # robot specs
    gripper_width = 0.02  # cm

    # depth filter, count valid object
    DEPTH_MIN = 0.01

    # workspace
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])

# colors
# 106 135 142
blue_lower = np.array([96, 85, 92], np.uint8)
blue_upper = np.array([116, 185, 192], np.uint8)
# 56 130 137
green_lower = np.array([48, 80, 87], np.uint8)
green_upper = np.array([64, 180, 187], np.uint8)
# 11  97 131
brown_lower = np.array([8, 57, 91], np.uint8)
brown_upper = np.array([14, 137, 171], np.uint8)
# 15 209 206
orange_lower = np.array([12, 159, 156], np.uint8)
orange_upper = np.array([18, 255, 255], np.uint8)
# 23 177 202
yellow_lower = np.array([20, 127, 152], np.uint8)
yellow_upper = np.array([26, 227, 252], np.uint8)
# 158, 148, 146 to 5 19 158
gray_lower = np.array([0, 0, 108], np.uint8)
gray_upper = np.array([15, 56, 208], np.uint8)
# rgb(217, 74, 76) to 0 168 217
red_lower = np.array([0, 118, 172], np.uint8)
red_upper = np.array([10, 218, 255], np.uint8)
# rgb(148, 104, 136) to 158  76 148
purple_lower = np.array([148, 26, 98], np.uint8)
purple_upper = np.array([167, 126, 198], np.uint8)
# rgb(101, 156, 151) to 87  90 156
cyan_lower = np.array([77, 40, 106], np.uint8)
cyan_upper = np.array([97, 140, 206], np.uint8)
# rgb(216, 132, 141) to 177  99 216
pink_lower = np.array([168, 49, 166], np.uint8)
pink_upper = np.array([187, 149, 255], np.uint8)
colors_lower = [
    blue_lower,
    green_lower,
    brown_lower,
    orange_lower,
    yellow_lower,
    gray_lower,
    red_lower,
    purple_lower,
    cyan_lower,
    pink_lower]
colors_upper = [
    blue_upper,
    green_upper,
    brown_upper,
    orange_upper,
    yellow_upper,
    gray_upper,
    red_upper,
    purple_upper,
    cyan_upper,
    pink_upper]

# resolution and padding resolution
heightmap_resolution = 0.002
resolution = 224
resolution_pad = 320
padding_width = int((resolution_pad - resolution) / 2)
resolution_crop = 60

# pre training values
PUSH_Q = 0.25
GRASP_Q = 0.5

# black backgroud sim
# color_mean = [0.0235, 0.0195, 0.0163]
# color_std = [0.1233, 0.0975, 0.0857]
# depth_mean = [0.0022]
# depth_std = [0.0089]

# random sim
color_mean = [0.0272, 0.0225, 0.0184]
color_std = [0.1337, 0.1065, 0.0922]
depth_mean = [0.0020]
depth_std = [0.0073]

# binary
binary_mean = [0.2236]
binary_std = [0.4167]
used_binary_mean = [0.0635, 0.0289]
used_binary_std = [0.2439, 0.1675]


total_obj = 5
