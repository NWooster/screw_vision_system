#!/usr/bin/env python3

"""

Nathan Wooster
Jan 2022
Python script to automatically tune screw location parameters.

"""

import cv2 as cv
import numpy as np
import random

# custom imports
import screw_location
import error


def tune(iterations=100):
    # open ground truth .txt file
    ground_truths = np.loadtxt("combined_screw_ground_truths.txt", delimiter=",")

    # parameter ranges
    dp_low = 0.01
    dp_high = 2
    param1_low = 20
    param1_high = 150
    param2_low = 10
    param2_high = 100
    blue_t_low = 0
    blue_t_high = 150
    green_t_low = 0
    green_t_high = 150
    red_t_low = 0
    red_t_high = 150

    for i in range(iterations):
        # make parameters random number in given range
        dp = round(random.uniform(dp_low, dp_high), 2)
        param1 = random.randint(param1_low, param1_high)
        param2 = random.randint(param2_low, param2_high)
        blue_t = random.randint(blue_t_low, blue_t_high)
        green_t = random.randint(green_t_low, green_t_high)
        red_t = random.randint(red_t_low, red_t_high)

        # find pixel screw location
        pix_screw_locations = screw_location.pixel_screw_location(dp, param1, param2, image_location='images_taken/'
                                                                                                     '1latest_image_'
                                                                                                     'from_camera.jpg')
        print(dp, param1, param2)

        # find error
        screw_centres_found = pix_screw_locations[:, :2]  # select centres only for pixel location estimates
        e_t = error.total_error(screw_centres_found, ground_truths)
        print('total error:', e_t)

        # if error is less than previous save values


if __name__ == "__main__":
    tune(iterations=5)
