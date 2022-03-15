#!/usr/bin/env python3

"""

Nathan Wooster
Jan 2022
Python script to automatically tune screw location parameters.

"""

import math
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
    f_dp = 0
    f_param1 = 0
    f_param2 = 0
    f_fp = no_fp
    f_fn = no_fn
    f_correct = no_correct
    f_e_loc = e_loc
    f_blue = 0
    f_green = 0
    f_red = 0

    for i in range(iterations):

        # refine parameter ranges based off iteration number
        if i < iterations/2:
            # parameter ranges
            dp_low = 0.01
            dp_high = 2.5
            param1_low = 10
            param1_high = 140
            param2_low = 10
            param2_high = 120
            blue_t_low = 0
            blue_t_high = 255
            green_t_low = 0
            green_t_high = 255
            red_t_low = 0
            red_t_high = 255
        else:
            # refined parameter ranges
            range_mult = 0.25  # % change from current final parameter (range will be double this x current value)
            dp_low = round(f_dp - f_dp*range_mult, 2)
            dp_high = round(f_dp + f_dp*range_mult, 2)
            param1_low = round(f_param1 - f_param1*range_mult)
            param1_high = round(f_param1 + f_param1*range_mult)
            param2_low = round(f_param2 - f_param2*range_mult)
            param2_high = round(f_param2 + f_param2*range_mult)
            blue_t_low = round(f_blue - f_blue*range_mult)
            blue_t_high = round(f_blue + f_blue*range_mult)
            green_t_low = round(f_green - f_green*range_mult)
            green_t_high = round(f_green + f_green*range_mult)
            red_t_low = round(f_red - f_red*range_mult)
            red_t_high = round(f_red + f_red*range_mult)

        # make parameters random number in given range
        dp = round(random.uniform(dp_low, dp_high), 2)
        param1 = random.randint(param1_low, param1_high)
        param2 = random.randint(param2_low, param2_high)
        blue_t = random.randint(blue_t_low, blue_t_high)
        green_t = random.randint(green_t_low, green_t_high)
        red_t = random.randint(red_t_low, red_t_high)

        # find pixel screw location
        pix_screw_locations = screw_location.pixel_screw_location(dp, param1, param2, blue_t, green_t, red_t,
                                                                  image_location='images_taken/'
                                                                                 '1latest_image_from_camera.jpg')

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
        print(i, 'iterations completed')

    print('There are', f_fp, 'screws falsely labelled,', f_fn, 'screws that were missed and',
          f_correct, 'correctly found.')
    print('The location error of correctly found screws is:', f_e_loc, 'pixels.')
    print('')
    print('final error:', f_error)
    print('')
    print('optimised parameters: dp =', f_dp, '  param1 =', f_param1, '  param2=', f_param2,
          ' blue=', f_blue, ' green=', f_green, ' red=', f_red)

    # plot iterations vs error
    itera = np.arange(1, iterations+1)
    sns.set()
    plt.plot(itera, error_array)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    tune(iterations=50)
