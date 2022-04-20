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
import screw_location_tuning_test
import error


def tune(iterations=100, no_of_pics=10):
    # initialise values and arrays:

    # empty arrays to store info on each pic within 1 iteration
    e_total_array_pics = []
    fp_array_pics = []
    fn_array_pics = []
    no_correct_array_pics = []
    e_loc_array_pics = []

    # final average across all pics in one iteration
    # error array
    error_array = []
    # false pos array
    fp_array = []
    # false neg array
    fn_array = []
    # pixel loc error array
    pix_loc_err = []

    # returned values from error function
    f_error = math.inf
    no_fp = math.inf
    no_fn = math.inf
    no_correct = 0
    e_loc = math.inf

    # initialise final optimised values
    f_dp = 0
    f_param1 = 0
    f_param2 = 0
    f_fp = no_fp
    f_fn = no_fn
    f_correct = no_correct
    no_correct_max = no_correct
    no_correct_min = no_correct
    correct_std = math.inf
    f_correct_max = no_correct_max
    f_correct_min = no_correct_min
    f_correct_std = correct_std
    f_e_loc = e_loc

    f_blue_upper = 255
    f_blue_bottom = 0
    f_green_upper = 255
    f_green_bottom = 0
    f_red_upper = 255
    f_red_bottom = 0

    for i in range(iterations):

        # refine parameter ranges based off iteration number
        if i < iterations / 2:
            # parameter ranges
            dp_low = 1  # default 0.01
            dp_high = 2  # default 2.5
            param1_low = 5  # default 10
            param1_high = 50  # default 140
            param2_low = 10  # default 10
            param2_high = 80  # default 120

            # for colour upper must be greater than bottom so t_upper_low is given by t_bottom+2
            # blue_t_upper_low = 10  # not needed as specified by the lowest upper value
            blue_t_upper_high = 255  # default 255
            blue_t_bottom_low = 0  # default 0
            blue_t_bottom_high = 200  # default 250

            # green_t_upper_low = 10  # default 10
            green_t_upper_high = 255  # default 255
            green_t_bottom_low = 0  # default 0
            green_t_bottom_high = 200  # default 250

            # red_t_upper_low = 10  # default 10
            red_t_upper_high = 255  # default 255
            red_t_bottom_low = 0  # default 0
            red_t_bottom_high = 200  # default 250

        elif i >= iterations / 2:
            # refined parameter ranges
            range_mult = 0.25  # % 0.25 change from current final parameter (range will be double this x current value)
            dp_low = round(f_dp - f_dp * range_mult, 2)
            dp_high = round(f_dp + f_dp * range_mult, 2)
            param1_low = round(f_param1 - f_param1 * range_mult)
            param1_high = round(f_param1 + f_param1 * range_mult)
            param2_low = round(f_param2 - f_param2 * range_mult)
            param2_high = round(f_param2 + f_param2 * range_mult)

            # blue_t_upper_low = round(f_blue_upper - f_blue_upper * range_mult) # mot needed as
            # specified by bottom
            blue_t_upper_high = round(f_blue_upper + f_blue_upper * range_mult)
            blue_t_bottom_low = round(f_blue_bottom - f_blue_bottom * range_mult)
            blue_t_bottom_high = round(f_blue_bottom + f_blue_bottom * range_mult)

            # green_t_upper_low = round(f_green_upper - f_green_upper * range_mult)
            green_t_upper_high = round(f_green_upper + f_green_upper * range_mult)
            green_t_bottom_low = round(f_green_bottom - f_green_bottom * range_mult)
            green_t_bottom_high = round(f_green_bottom + f_green_bottom * range_mult)

            # red_t_upper_low = round(f_red_upper - f_red_upper * range_mult)
            red_t_upper_high = round(f_red_upper + f_red_upper * range_mult)
            red_t_bottom_low = round(f_red_bottom - f_red_bottom * range_mult)
            red_t_bottom_high = round(f_red_bottom + f_red_bottom * range_mult)

        # make parameters random number in given range
        dp = round(random.uniform(dp_low, dp_high), 2)
        param1 = random.randint(param1_low, param1_high)
        param2 = random.randint(param2_low, param2_high)

        # for colour upper must be greater than bottom so t_upper_low is given by t_bottom+2
        blue_t_bottom = random.randint(blue_t_bottom_low, blue_t_bottom_high)
        blue_t_upper_low = blue_t_bottom + 2
        # getting odd error sometimes
        blue_t_upper = random.randint(blue_t_upper_low, blue_t_upper_high)

        green_t_bottom = random.randint(green_t_bottom_low, green_t_bottom_high)
        green_t_upper_low = green_t_bottom + 2
        # getting odd error sometimes
        green_t_upper = random.randint(green_t_upper_low, green_t_upper_high)

        red_t_bottom = random.randint(red_t_bottom_low, red_t_bottom_high)
        red_t_upper_low = red_t_bottom + 2
        red_t_upper = random.randint(red_t_upper_low, red_t_upper_high)

        # CHANGE THESE TO FINAL PARAMS
        if i > -1:
            dp = 1.8
            param1 = 27
            param2 = 61
            blue_t_upper = 251
            blue_t_bottom = 103
            green_t_upper = 127
            green_t_bottom = 105
            red_t_upper = 129
            red_t_bottom = 54

        # iterate through images 1 to n (put n+1)
        number_of_pics = no_of_pics
        for n in range(1, number_of_pics + 1):

            ## tuning set
            #current_image = 'phone_pic' + str(n)
            ## testing set
            current_image = 'phone_picTest' + str(n)

            # find pixel screw location
            pix_screw_locations = screw_location_tuning_test.pixel_screw_location(dp, param1, param2, blue_t_upper,
                                                                             blue_t_bottom, green_t_upper,
                                                                             green_t_bottom, red_t_upper,
                                                                             red_t_bottom, picture=current_image,
                                                                             fast=1)
            # open ground truth .txt file
            ## tuning set
            #ground_truths = np.loadtxt('images_processed/' + current_image + '/combined_screw_ground_truths.txt', delimiter=",")
            ## testing set
            ground_truths = np.loadtxt('images_processed/TestSet/' + current_image + '/combined_screw_ground_truths.txt', delimiter=",")

            # check some screws exist
            if not pix_screw_locations.any():
                e_total = math.inf
            # find error
            else:
                screw_centres_found = pix_screw_locations[:, :2]  # select centres only for pixel location estimates
                no_fp_1pic, no_fn_1pic, no_correct_1pic, e_loc_1pic, e_total_1pic = error.total_error(
                    screw_centres_found, ground_truths)

                # save error data for that image in array
                e_total_array_pics.append(e_total_1pic)
                fp_array_pics.append(no_fp_1pic)
                fn_array_pics.append(no_fn_1pic)
                no_correct_array_pics.append(no_correct_1pic)
                e_loc_array_pics.append(e_loc_1pic)

        # catch case where no screws were found and therefore error not calculated
        if len(e_total_array_pics) == 0:
            e_total = math.inf
        else:
            # calc error average across all n images
            e_total = average(e_total_array_pics)
            no_fp = average(fp_array_pics)
            no_fn = average(fn_array_pics)
            no_correct = average(no_correct_array_pics)
            no_correct_max = max(no_correct_array_pics)
            no_correct_min = min(no_correct_array_pics)
            correct_std = np.std(no_correct_array_pics)
            e_loc = average(e_loc_array_pics)

        # empty arrays to store info on each pic within 1 iteration
        e_total_array_pics = []
        fp_array_pics = []
        fn_array_pics = []
        no_correct_array_pics = []
        e_loc_array_pics = []

        # if error is less than previous save values
        if e_total < f_error:
            f_dp = dp
            f_param1 = param1
            f_param2 = param2
            f_blue_upper = blue_t_upper
            f_blue_bottom = blue_t_bottom
            f_green_upper = green_t_upper
            f_green_bottom = green_t_bottom
            f_red_upper = red_t_upper
            f_red_bottom = red_t_bottom

            f_error = e_total
            f_fp = no_fp
            f_fn = no_fn
            f_correct = no_correct
            f_correct_max = no_correct_max
            f_correct_min = no_correct_min
            f_correct_std = correct_std
            f_e_loc = e_loc

        # store past errors results to plot
        error_array.append(f_error)
        fp_array.append(f_fp)
        fn_array.append(f_fn)
        pix_loc_err.append(f_e_loc)
        print(i + 1, 'iterations completed',
              '| error:', f_error, 'correct:' + str(f_correct), 'fp:' + str(f_fp), 'fn:' + str(f_fn))
        print('b_upper:', f_blue_upper, 'b_bottom:', f_blue_bottom,
              'g_upper:', f_green_upper, 'g_bottom:', f_green_bottom,
              'r_upper:', f_red_upper, 'r_bottom:', f_red_bottom)
        print('')

    print('There are on average', f_fp, 'screws falsely labelled,', f_fn, 'screws that were missed and',
          f_correct, 'correctly found.')
    print('The number of correctly found screws ranges from', f_correct_min, 'to', f_correct_max, 'with a standard deviation of', f_correct_std, '.')
    print('The average location error of correctly found screws is:', f_e_loc, 'pixels.')
    print('')
    print('final error:', f_error)
    print('')
    print('optimised parameters: dp =', f_dp, '  param1 =', f_param1, '  param2=', f_param2,
          ' blue_upper=', f_blue_upper, ' blue_bottom=', f_blue_bottom,
          ' green_upper=', f_green_upper, ' green_bottom=', f_green_bottom,
          ' red_upper=', f_red_upper, ' red_bottom=', f_red_bottom)

    plot_type = 1  # change how plot outputs (1or2)

    if plot_type == 1:

        itera = np.arange(1, iterations + 1)
        error_array = np.array(error_array)
        fp_array = np.array(fp_array)
        fn_array = np.array(fn_array)

        # plot iterations vs error function, fp, fn, pixel error
        sns.set()
        fig, ax = plt.subplots()
        plt.plot(itera, error_array, 'r', label='Error function')
        plt.plot(itera, fp_array, '--c', label='False positives')
        plt.plot(itera, fn_array, '--b', label='False negatives')
        plt.plot(itera, pix_loc_err, '--g', label='Mean pixel error')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('images_processed/error_graph.png')  # save graph as png
        plt.show()

    elif plot_type == 2:
        # plot iterations vs error
        itera = np.arange(1, iterations + 1)
        sns.set()
        plt.plot(itera, error_array)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()
        # plot iterations vs fp and fn
        itera = np.arange(1, iterations + 1)
        sns.set()
        plt.plot(itera, fp_array, label='False positives')
        plt.plot(itera, fn_array, label='False negatives')
        plt.xlabel('Iteration')
        plt.ylabel('Number vision found')
        plt.legend()
        plt.show()
        # plot iterations vs pixel error
        itera = np.arange(1, iterations + 1)
        sns.set()
        plt.plot(itera, pix_loc_err)
        plt.xlabel('Iteration')
        plt.ylabel('Mean pixel error')
        plt.legend()
        plt.show()

    # output final images
    for n in range(1, number_of_pics + 1):

        ## tuning set
        #current_image = 'phone_pic' + str(n)
        #pix_screw_locations = screw_location_tuning.pixel_screw_location(f_dp, f_param1, f_param2,
        #                                                                 f_blue_upper, f_blue_bottom,
        #                                                                 f_green_upper, f_green_bottom,
        #                                                                 f_red_upper, f_red_bottom,
        #                                                                 picture=current_image, fast=0)

        # open ground truth .txt file
        #ground_truths = np.loadtxt('images_processed/' + current_image + '/combined_screw_ground_truths.txt',
        #                           delimiter=",")

        #error.draw_error(pix_screw_locations, ground_truths,
        #                 image_location=str('images_taken/ToTune/' + current_image + '.jpg'),
        #                 save_image='images_processed/' + current_image + '/error.jpg')

        ## testing set
        current_image = 'phone_picTest' + str(n)
        pix_screw_locations = screw_location_tuning_test.pixel_screw_location(f_dp, f_param1, f_param2,
                                                                         f_blue_upper, f_blue_bottom,
                                                                         f_green_upper, f_green_bottom,
                                                                         f_red_upper, f_red_bottom,
                                                                         picture=current_image, fast=0)

        # open ground truth .txt file
        ground_truths = np.loadtxt('images_processed/TestSet/' + current_image + '/combined_screw_ground_truths.txt',
                                   delimiter=",")

        error.draw_error(pix_screw_locations, ground_truths,
                         image_location=str('images_taken/ToTest/' + current_image + '.jpg'),
                         save_image='images_processed/TestSet/' + current_image + '/error.jpg')



def average(list):
    return sum(list) / len(list)


if __name__ == "__main__":
    tune(iterations=1, no_of_pics=8)
