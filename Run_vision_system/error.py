#!/usr/bin/env python3

"""

Nathan Wooster
Jan 2022
Python script to work out screw location pixel error.

"""

import cv2 as cv
import numpy as np
import math

# custom imports
import screw_location
import resize_to_fit_screen


# function to return average error between two sets of points
def location_error(estimate, ground_truth):
    """
    Function works out pixel location error between 2 sets of points.
    """

    # if number of screws is not right
    if np.shape(estimate) != np.shape(ground_truth):
        # print("WARNING: Error in removing false positives and negatives as the number of screws found:",
        # np.shape(estimate)[0],
        # ", does not match the number of actual screws:", np.shape(ground_truth)[0])
        not_important = 1
    elif np.shape(estimate) == np.shape(ground_truth):
        pass

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


# function to calculate the number of false positives and negatives
def false_pos_neg(estimate, ground_truth):
    """
    Function returns:
     - an array of the estimates with flag 1 if they are false positives (wrongly circled).
     - an array of the ground truths with flag 1 if they are false negatives (not found).
    """

    # threshold pixel distance for being false pos and false neg
    threshold = 25  # pixels (0.07*25=1.75mm)

    # initialise an array to store all the smallest distances FROM ESTIMATES
    small_dist_est = np.full((np.shape(estimate)[0], 1), np.inf)

    # initialise an array to store all the smallest distances FROM GROUND TRUTHS
    small_dist_gt = np.full((np.shape(ground_truth)[0], 1), np.inf)

    # initialise array to hold false pos and false neg info
    z1 = np.zeros((np.shape(estimate)[0], 1))
    estimate_flag = np.concatenate((estimate, z1), axis=1)
    z2 = np.zeros((np.shape(ground_truth)[0], 1))
    ground_truth_flag = np.concatenate((ground_truth, z2), axis=1)

    # calc smallest distance to known screw for each estimate and if larger than threshold label as a false pos
    for i in range(np.shape(estimate)[0]):
        for n in range(np.shape(ground_truth)[0]):
            current_dist = distance(estimate[i, 0], estimate[i, 1], ground_truth[n, 0], ground_truth[n, 1])
            if current_dist < small_dist_est[i]:
                small_dist_est[i] = current_dist
        if small_dist_est[i] > threshold:
            estimate_flag[i, 2] = 1

    # calc smallest distance to estimate for each known screw and if larger than threshold label as a false neg
    for n in range(np.shape(ground_truth)[0]):
        for i in range(np.shape(estimate)[0]):
            current_dist = distance(estimate[i, 0], estimate[i, 1], ground_truth[n, 0], ground_truth[n, 1])
            if current_dist < small_dist_gt[n]:
                small_dist_gt[n] = current_dist
        if small_dist_gt[n] > threshold:
            ground_truth_flag[n, 2] = 1

    # return list if estimates and ground truths with false pos and false neg flags
    return estimate_flag, ground_truth_flag


def total_error(estimate, ground_truth):
    """
    Function to return final error as combination
    of false positives, negatives and location error displacement.
    """

    # if number of screws is not right
    if np.shape(estimate) != np.shape(ground_truth):
        # print("The number of screws found:", np.shape(estimate)[0],
        #      ", does not match the number of actual screws:", np.shape(ground_truth)[0])
        not_important = 1
    elif np.shape(estimate) == np.shape(ground_truth):
        # print("The number of screws found:", np.shape(estimate)[0],
        #      ", does match the number of actual screws:", np.shape(ground_truth)[0])
        not_important = 1

    false_pos, false_neg = false_pos_neg(estimate, ground_truth)

    # calc number of false pos, false neg and correct detections
    no_fp = sum(false_pos[:, 2])
    no_fn = sum(false_neg[:, 2])
    no_correct = np.shape(estimate)[0] - no_fp

    # error check
    if np.shape(estimate)[0] - np.shape(ground_truth)[0] != no_fp - no_fn:
        # number of false pos - false neg should equal difference in estimated and actual screw count
        # print('PROBLEM WITH ERROR CALCULATION')
        not_important = 1

    # remove false pos and false neg from list
    correct_estimates = np.delete(false_pos, np.where(false_pos[:, 2] == 1)[0], 0)
    found_ground_truths = np.delete(false_neg, np.where(false_neg[:, 2] == 1)[0], 0)

    # find location error
    e_loc = location_error(correct_estimates, found_ground_truths)

    # calculate total error from false neg, pos and location
    fpos_weight = 1
    fneg_weight = 1.5
    e_total = (e_loc + (no_fp * fpos_weight) + (no_fn * fneg_weight))  # total error equation

    return no_fp, no_fn, no_correct, e_loc, e_total


def draw_error(estimate, ground_truth, image_location='images_taken/1latest_image_from_camera.jpg'):
    # select centres only for pixel location estimates
    screw_centres_found = estimate[:, :2]

    # loads an image
    image = cv.imread(cv.samples.findFile(image_location), cv.IMREAD_COLOR)

    # output false pos and false neg locations
    false_pos, false_neg = false_pos_neg(screw_centres_found, ground_truth)

    # draw false positives (wrongly labelled screws) as yellow circle
    false_pos_loc = np.delete(estimate, np.where(false_pos[:, 2] == 0)[0], 0)  # find false pos locations & radii
    for i in range(np.shape(false_pos_loc)[0]):
        image = cv.circle(image, np.uint16(false_pos_loc[i, (0, 1)]), radius=np.uint16(false_pos_loc[i, 2]),
                          color=(0, 255, 255), thickness=4)

    # draw false negatives (missed screws) as red circle
    false_neg_loc = np.delete(false_neg, np.where(false_neg[:, 2] == 0)[0], 0)  # find false neg locations
    for i in range(np.shape(false_neg_loc)[0]):
        image = cv.circle(image, np.uint16(false_neg_loc[i, (0, 1)]), radius=23,
                          color=(0, 0, 255), thickness=4)

    # draw correctly found estimates as pink circle
    correct_found_loc = np.delete(estimate, np.where(false_pos[:, 2] == 1)[0], 0)  # find locations & radii
    for i in range(np.shape(correct_found_loc)[0]):
        image = cv.circle(image, np.uint16(correct_found_loc[i, (0, 1)]), radius=np.uint16(correct_found_loc[i, 2]),
                          color=(255, 0, 255), thickness=3)

    # draw correctly found ground truths as blue dot
    found_gt_loc = false_neg
    for i in range(np.shape(found_gt_loc)[0]):
        image = cv.circle(image, np.uint16(found_gt_loc[i, (0, 1)]), radius=7,
                          color=(0, 255, 0), thickness=-1)

    # color key
    cv.putText(image, text="KEY", org=(1800, 1450), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=2, color=(0, 0, 0), thickness=3)
    cv.putText(image, text="False positive screws", org=(1800, 1500), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(0, 255, 255), thickness=2)
    cv.putText(image, text="Screws not found", org=(1800, 1550), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(0, 0, 255), thickness=2)
    cv.putText(image, text="Screws correctly found", org=(1800, 1600), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(255, 0, 255), thickness=2)
    cv.putText(image, text="Ground truths", org=(1800, 1650), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(0, 255, 0), thickness=2)

    # resize and show image
    resized_image = resize_to_fit_screen.resize(image, 1000)
    cv.imshow("screw error", resized_image)

    # save image as filename.jpeg
    cv.imwrite('images_processed/error' + '.jpg', image)

    cv.waitKey(0)  # wait till user exits or presses q

    return


# function to return distance between 2 points
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


if __name__ == "__main__":
    # find pixel screw location
    pix_screw_locations = screw_location.pixel_screw_location(image_location='images_taken/'
                                                                             '1latest_image_from_camera.jpg')

    # open ground truth .txt file
    ground_truths = np.loadtxt("combined_screw_ground_truths.txt", delimiter=",")

    # call error function (calculates number of false pos, neg and location error of correctly found screws
    screw_centres_found = pix_screw_locations[:, :2]  # select centres only for pixel location estimates
    no_fp, no_fn, no_correct, e_loc, e_total = total_error(screw_centres_found, ground_truths)

    # print outputs
    print('There are', no_fp, 'screws falsely labelled,', no_fn, 'screws that were missed and',
          no_correct, 'correctly found.')
    print('The location error of correctly found screws is:', e_loc, 'pixels.')
    print('')
    print('total error:', e_total)

    # draw the error visually
    draw_error(pix_screw_locations, ground_truths)
