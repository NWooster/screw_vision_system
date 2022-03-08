#!/usr/bin/env python3

'''

Nathan Wooster
Jan 2022
Python script to work out screw location pixel error.

'''

import cv2 as cv
import numpy as np
import math

# custom imports
import calibrate_camera
import screw_location


# function to return average error between two sets of points
def error(estimate, ground_truth):
    # if number of screws is not right
    if np.shape(estimate) != np.shape(ground_truth):
        print("WARNING: The number of screws found:", np.shape(estimate)[0],
              ", does not match the number of actual screws:", np.shape(ground_truth)[0])
    elif np.shape(estimate) == np.shape(ground_truth):
        print("The number of screws found:", np.shape(estimate)[0],
              ", does match the number of actual screws:", np.shape(ground_truth)[0])

    # initialise an array to store all the smallest distances to actual screws
    small_dist = np.full((np.shape(estimate)[0], 1), np.inf)

    # calc smallest distance to known screw for each estimate screw location and put in array
    for i in range(np.shape(estimate)[0]):
        for n in range(np.shape(ground_truth)[0]):
            current_dist = distance(estimate[i, 0], estimate[i, 1], ground_truth[n, 0], ground_truth[n, 1])
            if current_dist < small_dist[i]:
                small_dist[i] = current_dist

    print(small_dist)
    print()

    # calc mean error
    e = sum(small_dist) / np.shape(estimate)[0]

    print('mean error:', e, 'max error:', max(small_dist), 'min error:', min(small_dist))

    # convert to mm to check
    mm_to_pix = calibrate_camera.calibrate_camera(image_location='images_taken/1latest_image_from_camera.jpg')[0]
    print('mm of these:', e * mm_to_pix, max(small_dist) * mm_to_pix, min(small_dist * mm_to_pix))

    return e


# function to return distance between 2 points
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


if __name__ == "__main__":
    # find pixel screw location
    pix_screw_locations = screw_location.pixel_screw_location(image_location='images_taken/'
                                                                             '1latest_image_from_camera.jpg')

    # select centres only for pixel location estimates
    screw_centres_found = pix_screw_locations[:, :2]

    # open ground truth .txt file
    ground_truths = np.loadtxt("combined_screw_ground_truths.txt", delimiter=",")

    # re-order based on y coordinate (doesn't actually matter)
    screw_centres_found = screw_centres_found[screw_centres_found[:, 1].argsort()]
    ground_truths = ground_truths[ground_truths[:, 1].argsort()]

    # call error function
    error(screw_centres_found, ground_truths)
