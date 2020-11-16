import cv2
import sys
import math
import time
import random
import threading
import numpy as np
from pprint import pprint

import os
import json


def main(file_path):
    """
    Auto mask annotation by color.
    To use this script, you should measure the color of objects so that maaks can be extracted.
    """

    data = {}
    # Preset color threshold for each color
    color_threshold = {
        "yellow": {"low": [33, 31, 70], "high": [59, 135, 255]},
        "green": {"low": [76, 144, 110], "high": [90, 255, 230]},
        "blue": {"low": [90, 90, 110], "high": [140, 255, 255]},
        "orange": {"low": [0, 35, 117], "high": [13, 185, 255]},
        "red": {"low": [0, 180, 157], "high": [7, 255, 227]},
        "natural": {"low": [13, 4, 115], "high": [157, 57, 228]},
    }
    background_threshold = {"low": np.array([0, 0, 100], np.uint8), "high": np.array([255, 255, 255], np.uint8)}

    for key in color_threshold.keys():
        for sub_key in color_threshold[key].keys():
            color_threshold[key][sub_key] = np.array(color_threshold[key][sub_key], np.uint8)
    # Auto annotation for each img file
    for file_name in os.listdir(file_path):
        # Read image
        img_path = os.path.join(file_path, file_name)
        img = cv2.imread(img_path)
        # Pre-processing: noise removal + cvt color
        # img = cv2.medianBlur(img, 5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # # Remove background
        # bg_mask = cv2.inRange(img, color_threshold["background"]["low"], color_threshold["background"]["high"])
        regions = list()
        # Iterate through all colors to find objects
        for _color, threshold in color_threshold.items():
            # color_mask = cv2.inRange(img, threshold["low"], threshold["high"])
            mask = cv2.inRange(img, threshold["low"], threshold["high"])
            # Find contours and write annotation
            contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) > 200:
                    region = {"shape_attributes": {"name": "polygon", "all_points_x": c[:, 0, 0].tolist(
                    ), "all_points_y": c[:, 0, 1].tolist()}, "region_attributes": {}}
                    regions.append(region)
        # Write annotation for one img
        file_size = os.stat(img_path).st_size
        key = file_name + str(file_size)
        data[key] = {"filename": file_name, "size": file_size, "regions": regions, "file_attributes": {}}

        # # Remove background
        # bg_mask = cv2.inRange(img, background_threshold["low"], background_threshold["high"])
        # img = cv2.bitwise_and(img, img, mask=bg_mask)
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        # cv2.imwrite(img_path, img)
    # Dump json file
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    main(sys.argv[1])
