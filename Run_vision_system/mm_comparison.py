#!/usr/bin/env python3

"""

Nathan Wooster
Jan 2022
Python script to calculate mean mm accuracy between ground truths and estimates.

"""

import cv2 as cv
import numpy as np
import math


def mm_error(estimate, ground_truth):

    # initialise an array to store all the smallest distances to actual screws
    small_dist = np.full((np.shape(estimate)[0], 1), np.inf)

    # calc smallest distance to known screw for each estimate screw location and put in array
    for i in range(np.shape(estimate)[0]):
        for n in range(np.shape(ground_truth)[0]):
            current_dist = distance(estimate[i, 0], estimate[i, 1], ground_truth[n, 0], ground_truth[n, 1])
            if current_dist < small_dist[i]:
                small_dist[i] = current_dist

    # calc mean error (catch divide by 0 error)
    if np.shape(estimate)[0] == 0:
        e = math.inf
    else:
        e = sum(small_dist) / np.shape(estimate)[0]

    # print('mean error:', e, 'max error:', max(small_dist), 'min error:', min(small_dist))

    # convert to mm to check
    # mm_to_pix = calibrate_camera.calibrate_camera(image_location='images_taken/1latest_image_from_camera.jpg')[0]
    # print('mm of these:', e * mm_to_pix, max(small_dist) * mm_to_pix, min(small_dist * mm_to_pix))

    return e