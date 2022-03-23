#!/usr/bin/env python3

'''

Nathan Wooster
Jan 2022
Python script to work out screw locations.

'''

import cv2 as cv
import numpy as np
import math

# custom imports
import calibrate_camera


def mm_screw_location(pix_to_mm, origin_pix, ratio_error, image_location='images_taken/1latest_image_from_camera.jpg'):
    """
    Screw location function for mm coordinates.
    Calculates it from a given origin (normally top left corner of chessboard is specified).
    """

    # call pixel location function which returns screw centres and radii
    pix_locations = pixel_screw_location(image_location=image_location)

    # find pixel location with relation to specified origin (could be -ve as above origin)
    origin_pix = origin_pix.reshape(1, 2)  # reshape
    origin_pix = np.hstack((origin_pix, np.zeros((origin_pix.shape[0], 1), dtype=origin_pix.dtype)))  # add zero col
    pix_loc_from_origin = pix_locations - origin_pix  # take away to get screw pix location from origin

    # convert to mm coordinates
    mm_locations = pix_loc_from_origin * pix_to_mm

    # find max error in mm
    max_mm_error = np.amax(abs(mm_locations)) * ratio_error

    # find direct distance from origin in pixels and mm
    direct_pix = np.zeros((np.shape(pix_locations)[0], 1))
    direct_mm = np.zeros((np.shape(pix_locations)[0], 1))
    for i in range(np.shape(pix_locations)[0]):
        direct_pix[i] = distance(pix_locations[i, 0], pix_locations[i, 1], origin_pix[0, 0], origin_pix[0, 1])
        direct_mm[i] = direct_pix[i]*pix_to_mm
    direct_pix = np.round(direct_pix, 0)  # round to int
    direct_mm = np.round(direct_mm, 2)  # round to 2 decimal place

    # Generate image
    image = cv.imread(cv.samples.findFile('images_processed/1screw_pixel_output.jpg'), cv.IMREAD_COLOR)
    cv.circle(image, (int(origin_pix[0, 0]), int(origin_pix[0, 1])), 25, (0, 0, 0))  # draw circle at origin
    font = cv.FONT_HERSHEY_SIMPLEX  # set font
    # place origin text
    cv.putText(image, '(0,0)', (int(origin_pix[0, 0]), int(origin_pix[0, 1])), font, 1, (0, 0, 255), 2)
    # draw on mm and pixel location at each screw
    pix_loc_from_origin_rounded = pix_loc_from_origin[:, :3]
    mm_locations_rounded = mm_locations[:, :3]
    pix_loc_from_origin_rounded = np.round(pix_loc_from_origin_rounded, 1)  # round to 1 decimal place
    mm_locations_rounded = np.round(mm_locations_rounded, 1)  # round to 1 decimal place
    for i in range(np.shape(pix_locations)[0]):
        cv.putText(image, str(direct_pix[i]),
                   (int(pix_locations[i, 0]), int(pix_locations[i, 1]) - 50),
                   font, 0.7, (100, 0, 255), 2)
        cv.putText(image, str(direct_mm[i]),
                   (int(pix_locations[i, 0]), int(pix_locations[i, 1]) - 25),
                   font, 0.7, (0, 255, 255), 2)
        #cv.putText(image, str(pix_loc_from_origin_rounded[i]),
                   #(int(pix_locations[i, 0]), int(pix_locations[i, 1]) - 10),
                   #font, 0.7, (255, 255, 255), 2)
        cv.putText(image, str(mm_locations_rounded[i]), (int(pix_locations[i, 0]), int(pix_locations[i, 1]) + 15),
                   font, 0.7, (0, 0, 255), 2)

    # color key
    cv.putText(image, text="KEY", org=(1600, 1450), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=2, color=(0, 0, 0), thickness=3)
    cv.putText(image, text="Direct distance in pixels from origin", org=(1600, 1500), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(100, 0, 255), thickness=2)
    cv.putText(image, text="Direct distance in mm from origin", org=(1600, 1550), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(0, 255, 255), thickness=2)
    cv.putText(image, text="[X, y, r] in pixels from origin", org=(1600, 1600), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(255, 255, 255), thickness=2)
    cv.putText(image, text="[X, y, r] in mm from origin", org=(1600, 1650), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(0, 0, 255), thickness=2)

    # save image
    cv.imwrite('images_processed/1screw_mm_output' + '.jpg', image)

    return mm_locations, max_mm_error


# loads image, pre-process it, apply hough circle detection
def pixel_screw_location(dp=1.49, param1=17, param2=47, blue_t=107, green_t=362, red_t=58,
                         image_location='images_taken/1latest_image_from_camera'):
    """
    Screw location function for pixel coordinates.
    """

    # labels where image is
    filename = image_location

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
    min_dist = int(min_r * 2)  # min distance between two screws
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

            # change flag
            if blue < blue_thresh and green < green_thresh and red < red_thresh:
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
    cv.imwrite('images_processed/1screw_pixel_output' + '.jpg', final_image)

    return screw_locations


# function to return distance between 2 points
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


if __name__ == "__main__":

    pix_to_mm, tl_corner_pix, ratio_error = calibrate_camera.calibrate_camera(image_location='images_taken/'
                                                                                             '1latest_image_from_'
                                                                                             'camera.jpg')

    # test with different mm ratios (normally get from calibrate camera)
    #mm_ratio = 0.0579  # (from notes calc)
    #mm_ratio = 0.05765  # (from average of 3 screws notes calc)
    #mm_ratio = 0.07446754918921591  # from large rectangle cheq square
    #mm_ratio = 0.05741293532  # (from top right screw)
    mm_ratio = 0.05718463  # (from top right screw)

    pix_to_mm = mm_ratio

    print('ratio is:' + str(pix_to_mm))

    mm_screw_location(pix_to_mm, tl_corner_pix, ratio_error,
                      image_location='images_taken/1latest_image_from_camera.jpg')
