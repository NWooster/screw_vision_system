#!/usr/bin/env python3

'''

Nathan Wooster
Jan 2022
Python script to tune Hough Circle Parameters

'''

import cv2 as cv
import numpy as np
import math


# loads image, pre-process it, apply hough circle detection
def pixel_screw_location(dp, param1, param2, blue_t, green_t, red_t,
                         picture):
    """
    Screw location function for pixel coordinates.
    """

    # labels where image is
    filename = str('images_taken/ToTune/' + picture + '.jpg')

    # loads an image and calls it 'initial_image'
    initial_image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # check if image is loaded fine
    if initial_image is None:
        print('Error opening image!')
        return -1

    # convert image to grayscale from BGR and new image called 'gray'
    gray = cv.cvtColor(initial_image, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray image', gray)

    # adds medium blur to image to reduce noise (avoids false circle detection)
    blur_image = cv.medianBlur(gray, 5)
    # blur_resized = resize(blur_image, 600)
    # cv.imshow('Blur image', blur_resized)

    # numpy array .shape[0] outputs the number of elements in dimension 1 of the array (number of pixel rows)
    rows = blur_image.shape[0]
    # print(rows)
    # print(blur_image)

    '''
    Hough circle algorithm arguments:

    circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
    image: Input image (grayscale and blurred).
    HOUGH_GRADIENT: Define the detection method. One method available in OpenCV.
    dp: The inverse ratio of resolution.
    min_dist: Minimum distance between detected centers.
    param_1: Upper threshold for the internal Canny edge detector.
    param_2: Threshold for center detection.
    min_radius: Minimum radius to be detected. If unknown, put zero as default.
    max_radius: Maximum radius to be detected. If unknown, put zero as default.
    '''

    # parameters for Hough Circle algorithm
    dp = dp  # high dp means low matrix resolution so takes circles that do not have clear boundary (default 1)
    min_r = 18  # min pixel radius of screw (default 18)
    max_r = 30  # max pixel radius of screw (default 30)
    min_dist = int(min_r * 1.5)  # min distance between two screws
    param1 = param1  # if low then more weak edges will be found so weak circles returned (default 60)
    param2 = param2  # if low then more circles will be returned by HoughCircles (default 30)

    # apply OpenCV HoughCircle algorithm
    circles = cv.HoughCircles(blur_image, cv.HOUGH_GRADIENT, dp, min_dist,
                              param1=param1, param2=param2,
                              minRadius=min_r, maxRadius=max_r)

    # get screw centres and radius into np.array
    screw_locations = np.array(circles)
    screw_locations = np.squeeze(screw_locations, axis=0)  # remove redundant dimension

    # flag to run optimise parts
    optimised = 1

    blue_thresh = blue_t
    green_thresh = green_t
    red_thresh = red_t

    # check if wanting to colour optimise and screw array is not empty
    if optimised == 1 and screw_locations.size != 1:

        # add column to show think it is false positive
        z1 = np.zeros((np.shape(screw_locations)[0], 1))
        screw_locations = np.concatenate((screw_locations, z1), axis=1)

        # flag false pos by checking colour
        for i in range(np.shape(screw_locations)[0]):
            pix_check_x = int(screw_locations[(i, 0)])  # grab x coord
            pix_check_y = int(screw_locations[(i, 1)])  # grab y coord

            # catch error of pixel being out of bounds (not sure why this error happens as pixel 1944x2592 shld exist)
            if pix_check_y > 1943:
                pix_check_y = 1943
                print('ERROR IN Y PIXEL LOCATION')
            if pix_check_x > 2591:
                print('ERROR IN X PIXEL LOCATION')
                pix_check_x = 2591

            # find colours
            blue = initial_image[pix_check_y, pix_check_x, 0]
            green = initial_image[pix_check_y, pix_check_x, 1]
            red = initial_image[pix_check_y, pix_check_x, 2]

            # change false pos flag
            # colour (default is 'and' statements)
            if blue < blue_thresh and green < green_thresh and red < red_thresh:
                screw_locations[i, 3] = 1
            # y location not on phone
            if pix_check_y > 1290:
                screw_locations[i, 3] = 1

        # remove all flagged presumed false positives and remove added flag column
        screw_locations = np.delete(screw_locations, np.where(screw_locations[:, 3] == 1)[0], 0)
        screw_locations = np.delete(screw_locations, np.s_[-1:], axis=1)

    # initialise final image
    final_image = initial_image

    # draw the detected circles
    if circles is not None:
        # removes decimals
        circles_draw = np.uint16(np.around(screw_locations))
        # print('circles drawn:', circles_draw)
        for i in circles_draw:
            center = (i[0], i[1])
            # circle center
            cv.circle(final_image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            # draws circles (r,b,g) colour
            cv.circle(final_image, center, radius, (255, 0, 255), 3)

    # call imported resize image function specify required width (default 600)
    # resized_image = resize(final_image, 600)
    # show resized image
    # cv.imshow("detected screws", resized_image)

    # save image as filename.jpeg
    cv.imwrite('images_processed/' + picture + '/' + 'screw_pixel_output.jpg', final_image)

    return screw_locations


# function to return distance between 2 points
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


if __name__ == "__main__":
    pass
