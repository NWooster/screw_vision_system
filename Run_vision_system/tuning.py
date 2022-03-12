#!/usr/bin/env python3

"""

Nathan Wooster
Jan 2022
Python script to automatically tune screw location parameters.

"""

import math
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

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

    # initialise values:
    # error array
    error_array = []
    # returned values from error function
    f_error = math.inf
    no_fp = math.inf
    no_fn = math.inf
    no_correct = 0
    e_loc = math.inf
    # final optimised values
    f_dp = dp_low
    f_param1 = param1_low
    f_param2 = param1_low
    f_fp = no_fp
    f_fn = no_fn
    f_correct = no_correct
    f_e_loc = e_loc

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

        # check some screws exist
        if not pix_screw_locations.any():
            e_total = math.inf
        # find error
        else:
            screw_centres_found = pix_screw_locations[:, :2]  # select centres only for pixel location estimates
            no_fp, no_fn, no_correct, e_loc, e_total = error.total_error(screw_centres_found, ground_truths)

        # if error is less than previous save values
        if e_total < f_error:
            f_dp = dp
            f_param1 = param1
            f_param2 = param2
            f_blue = blue_t
            f_green = green_t
            f_red = red_t
            f_error = e_total
            f_fp = no_fp
            f_fn = no_fn
            f_correct = no_correct
            f_e_loc = e_loc

        # store past errors to plot
        error_array.append(f_error)
        print(i, 'completed')

    print('There are', f_fp, 'screws falsely labelled,', f_fn, 'screws that were missed and',
          f_correct, 'correctly found.')
    print('The location error of correctly found screws is:', f_e_loc, 'pixels.')
    print('')
    print('final error:', f_error)
    print('')
    print('optimised parameters: dp =', f_dp, '  param1 =', f_param1, '  param 2=', f_param2)

    # plot iterations vs error
    itera = np.arange(1, iterations+1)
    sns.set()
    plt.plot(itera, error_array)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    tune(iterations=10)
