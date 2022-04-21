#!/usr/bin/env python3

"""

Nathan Wooster
Jan 2022
Python script to calculate mean mm accuracy between ground truths and estimates.

"""

import cv2 as cv
import numpy as np
import math


# loads image, pre-process it, apply hough circle detection
def pixel_screw_location(dp=1.8, param1=27, param2=61, blue_t_upper=251, blue_t_bottom=103, green_t_upper=127,
                         green_t_bottom=105, red_t_upper=129, red_t_bottom=54, picture='phone_picTest1', fast=0):
    """
    Screw location function for pixel coordinates.
    """

    # labels where image is
    ## tuning set
    # filename = str('images_taken/ToTune/' + picture + '.jpg')
    ## testing set
    filename = str('images_taken/ToTest/' + picture + '.jpg')

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
                # print('ERROR IN Y PIXEL LOCATION')
                # print('pixel out of 1944 bound:', pix_check_y)
                pix_check_y = 1943
            if pix_check_x > 2591:
                # print('ERROR IN X PIXEL LOCATION')
                # print('pixel out of 2592 bound:', pix_check_x)
                pix_check_x = 2591

            # find colours
            blue = initial_image[pix_check_y, pix_check_x, 0]
            green = initial_image[pix_check_y, pix_check_x, 1]
            red = initial_image[pix_check_y, pix_check_x, 2]

            # change false pos flag
            # colour (default is 'and' statements)
            if (blue < blue_t_bottom or blue > blue_t_upper) and (green < green_t_bottom or green > green_t_upper) and (
                    red < red_t_bottom or red > red_t_upper):
                screw_locations[i, 3] = 1  # if in this colour bound set as fp
            # y location not on phone
            if pix_check_y > 1290:
                screw_locations[i, 3] = 1

        # remove all flagged presumed false positives and remove added flag column
        screw_locations = np.delete(screw_locations, np.where(screw_locations[:, 3] == 1)[0], 0)
        screw_locations = np.delete(screw_locations, np.s_[-1:], axis=1)

    # initialise final image
    final_image = initial_image

    # if wanting to print image
    if fast == 0:
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
        ## tuning set
        # cv.imwrite('images_processed/' + picture + '/' + 'screw_pixel_output.jpg', final_image)
        ## testing set
        cv.imwrite(str('images_processed/TestSet/' + picture + '/' + 'screw_pixel_output.jpg'), final_image)

    return screw_locations


def mm_screw_location(pix_to_mm, origin_pix, picture):
    """
    Screw location function for mm coordinates.
    Calculates it from a given origin (normally top left corner of chessboard is specified).
    """

    # call pixel location function which returns screw centres and radii
    pix_locations = pixel_screw_location(picture=picture)

    # find pixel location with relation to specified origin (could be -ve as above origin)
    origin_pix = origin_pix.reshape(1, 2)  # reshape
    origin_pix = np.hstack((origin_pix, np.zeros((origin_pix.shape[0], 1), dtype=origin_pix.dtype)))  # add zero col
    pix_loc_from_origin = pix_locations - origin_pix  # take away to get screw pix location from origin

    # convert to mm coordinates
    mm_locations = pix_loc_from_origin * pix_to_mm

    # find direct distance from origin in pixels and mm
    direct_pix = np.zeros((np.shape(pix_locations)[0], 1))
    direct_mm = np.zeros((np.shape(pix_locations)[0], 1))
    for i in range(np.shape(pix_locations)[0]):
        direct_pix[i] = distance(pix_locations[i, 0], pix_locations[i, 1], origin_pix[0, 0], origin_pix[0, 1])
        direct_mm[i] = direct_pix[i] * pix_to_mm

    # order screws by closest distance to origin:
    # pixel arrays
    pix_locations = np.concatenate((pix_locations, direct_pix), axis=1)  # put direct_mm array on end of mm_locations
    pix_locations = pix_locations[pix_locations[:, 3].argsort()]  # re-order based on direct_mm
    pix_locations = np.delete(pix_locations, -1, axis=1)  # delete last column

    pix_loc_from_origin = np.concatenate((pix_loc_from_origin, direct_pix), axis=1)  # put array on end
    pix_loc_from_origin = pix_loc_from_origin[pix_loc_from_origin[:, 3].argsort()]  # re-order based on direct_mm
    pix_loc_from_origin = np.delete(pix_loc_from_origin, -1, axis=1)  # delete last column

    direct_pix = np.sort(direct_pix, axis=0)  # order 1D array

    # mm array
    mm_locations = np.concatenate((mm_locations, direct_mm), axis=1)  # put direct_mm array on end of mm_locations
    mm_locations = mm_locations[mm_locations[:, 3].argsort()]  # re-order based on direct_mm
    mm_locations = np.delete(mm_locations, -1, axis=1)  # delete last column

    direct_mm = np.sort(direct_mm, axis=0)  # order 1D array

    # round direct_pix and mm
    direct_pix = np.round(direct_pix, 0)  # round to int
    direct_mm = np.round(direct_mm, 2)  # round to 2 decimal place

    # Generate image
    filename = str('images_processed/TestSet/' + picture + '/' + 'screw_pixel_output.jpg')
    image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    cv.circle(image, (int(origin_pix[0, 0]), int(origin_pix[0, 1])), 25, (0, 0, 255), thickness=2)  # draw origin
    font = cv.FONT_HERSHEY_SIMPLEX  # set font
    # place origin text
    cv.putText(image, 'Origin', (int(origin_pix[0, 0] + 30), int(origin_pix[0, 1] - 20)), font, 1.6, (0, 0, 255), 2)
    # draw on mm and pixel location at each screw
    pix_loc_from_origin_rounded = pix_loc_from_origin[:, :3]
    mm_locations_rounded = mm_locations[:, :3]
    pix_loc_from_origin_rounded = np.round(pix_loc_from_origin_rounded, 1)  # round to 1 decimal place
    mm_locations_rounded = np.round(mm_locations_rounded, 1)  # round to 1 decimal place
    for i in range(np.shape(pix_locations)[0]):
        cv.putText(image, str(i),
                   (int(pix_locations[i, 0]) - 50, int(pix_locations[i, 1]) - 30),
                   font, 1.2, (0, 255, 255), 2)  # default colour (50, 205, 50), 2)
        # cv.putText(image, str(direct_pix[i]),
        #           (int(pix_locations[i, 0]), int(pix_locations[i, 1]) - 50),
        #           font, 0.7, (100, 0, 255), 2)
        # cv.putText(image, str(direct_mm[i]),
        #           (int(pix_locations[i, 0]), int(pix_locations[i, 1]) - 25),
        #           font, 0.7, (0, 255, 255), 2)
        # cv.putText(image, str(pix_loc_from_origin_rounded[i]),
        #           (int(pix_locations[i, 0]), int(pix_locations[i, 1]) - 10),
        #           font, 0.7, (255, 255, 255), 2)
        cv.putText(image, str(mm_locations_rounded[i]), (int(pix_locations[i, 0]), int(pix_locations[i, 1]) - 15),
                   font, 0.7, (0, 0, 255), 2)

    # color key
    cv.putText(image, text="KEY", org=(1600, 1450), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=2, color=(0, 0, 0), thickness=3)
    cv.putText(image, text="Screw number (starts at 0)", org=(1600, 1500), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(0, 255, 255), thickness=2)
    # cv.putText(image, text="Direct distance in pixels from origin", org=(1600, 1550), fontFace=cv.FONT_HERSHEY_DUPLEX,
    #           fontScale=1.5, color=(100, 0, 255), thickness=2)
    # cv.putText(image, text="Direct distance in mm from origin", org=(1600, 1600), fontFace=cv.FONT_HERSHEY_DUPLEX,
    #           fontScale=1.5, color=(0, 255, 255), thickness=2)
    # cv.putText(image, text="[X, y, r] in pixels from origin", org=(1600, 1650), fontFace=cv.FONT_HERSHEY_DUPLEX,
    #           fontScale=1.5, color=(255, 255, 255), thickness=2)
    cv.putText(image, text="[X, y, r] in mm from origin", org=(1600, 1560), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1.5, color=(0, 0, 255), thickness=2)  # was org=(1600, 1700)

    # save image
    cv.imwrite(str('images_processed/TestSet/' + picture + '/' + '1screw_mm_output.jpg'), image)

    return mm_locations


def calibrate_camera(picture, mm_dist=80,
                     columns=7, rows=7):
    """
        `columns` and `rows` are the number of INSIDE corners in the
        chessboard's columns and rows.
        'mm_dist' is the distance in mm between each red dot.
    """

    # load image
    filename = str('images_taken/ToTest/' + picture + '.jpg')  # image location
    img = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # find corners
    ret, found_corners = cv.findChessboardCorners(gray, (columns, rows),
                                                  flags=cv.CALIB_CB_FAST_CHECK)

    # corners not found
    if ret != 1:
        print('chess board corners not found!')
        return 0

    # corners found
    else:

        # sub pixel adjustment algorithm
        """""
        cornersSubPix(image, corners, winSize, zeroZone, criteria)
        image: start image
        corners: old corner location before refining
        winSize: half size of search window
        zeroZone: (-1,-1) says no zero zone (no possible singularities)
        criteria: when the algorithm will exit
        - EPS: corner moves by this epsilon over 2 iterations the required accuracy is reached (default 0.001)
        - ITER: algorithm iterates this max amount (default 30)
        """
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        corners_sub_pix = cv.cornerSubPix(gray, found_corners, (5, 5), (-1, -1), criteria)

        # corner location and pixel array
        corner_location = np.array(corners_sub_pix)  # starts at top, left to right like pixels
        corner_location = np.squeeze(corner_location, axis=1)  # remove redundant array dimension
        pixel_size = img.shape[:2]
        # print('image size x:', pixel_size[1], 'y:', pixel_size[0])  # longer x creates rectangle

        # select 4 corners
        corner1 = corner_location[columns * rows - rows, :]
        corner2 = corner_location[0, :]
        corner3 = corner_location[columns - 1, :]
        corner4 = corner_location[columns * rows - 1, :]
        four_corners = np.array([corner1, corner2, corner3, corner4])

        # identify which corner is top left, top right, bottom left, bottom right
        four_corners = four_corners[four_corners[:, 1].argsort()]  # re-order based on y coordinate
        if four_corners[0, 0] < four_corners[1, 0]:
            tl = four_corners[0, :]
            tr = four_corners[1, :]
        else:
            tl = four_corners[1, :]
            tr = four_corners[0, :]

        if four_corners[2, 0] < four_corners[3, 0]:
            bl = four_corners[2, :]
            br = four_corners[3, :]
        else:
            bl = four_corners[3, :]
            br = four_corners[2, :]

        # check assigned corners correctly
        if tl[1] < bl[1] and tl[0] < tr[0] and tr[1] < br[1] and br[0] > bl[0]:
            # everything correct
            pass
        else:
            print('error corners not in the right location')

        # circle 4 corners (B,G,R) (top left corner should be black)
        tl_corner_rnd = np.rint(tl)  # round pixel float to nearest integer
        cv.circle(img, (int(tl_corner_rnd[0]), int(tl_corner_rnd[1])), 50, (0, 0, 0), 10)  # tl is BLACK

        tr_corner_rnd = np.rint(tr)  # round pixel float to nearest integer
        cv.circle(img, (int(tr_corner_rnd[0]), int(tr_corner_rnd[1])), 50, (0, 0, 255), 10)  # tr is RED

        bl_corner_rnd = np.rint(bl)  # round pixel float to nearest integer
        cv.circle(img, (int(bl_corner_rnd[0]), int(bl_corner_rnd[1])), 50, (0, 255, 0), 10)  # bl is GREEN

        br_corner_rnd = np.rint(br)  # round pixel float to nearest integer
        cv.circle(img, (int(br_corner_rnd[0]), int(br_corner_rnd[1])), 50, (255, 255, 0), 10)  # br is LIGHT BLUE

        # call function to get red dot pixel location
        red_dot_pix = find_red_dot(image_location=filename, picture=picture)

        # find red dot pix distance from each other
        red_dot_dist = distance(red_dot_pix[0, 0], red_dot_pix[0, 1], red_dot_pix[1, 0], red_dot_pix[1, 1])

        # calc mm to pixel ratio
        pix_mm_ratio = mm_dist / red_dot_dist

        # draw corners onto image
        cv.drawChessboardCorners(img, (columns, rows), corners_sub_pix, ret)
        # resized_image = resize(img, 800)  # resize image to fit screen
        # save non-resized image
        cv.imwrite(str('images_processed/TestSet/' + picture + '/' + '1chequered_cal.jpg'), img)

        # return top left corner coordinates in pixels
        tl_corner_pix = np.array(tl)

        # print()
        # print('press "q" to exit')
        # show image until user presses 'q'
        # while True:
        #    cv.imshow('found corners', resized_image)
        #    # press 'q' button to exit image
        #    if cv.waitKey(1) & 0xFF == ord('q'):
        #        break

    return pix_mm_ratio, tl_corner_pix


def find_red_dot(image_location, picture):
    # load image
    filename = image_location  # image location
    initial_image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # check if image is loaded fine
    if initial_image is None:
        print('Error opening image!')
        return -1

    gray = cv.cvtColor(initial_image, cv.COLOR_BGR2GRAY)
    blur_image = cv.medianBlur(gray, 5)

    # parameters for Hough Circle algorithm
    dp = 1  # high dp means low matrix resolution so takes circles that do not have clear boundary (default 1)
    min_r = 40  # min pixel radius of screw (default 40)
    max_r = 60  # max pixel radius of screw (default 60)
    min_dist = int(min_r * 2)  # min distance between two screws
    param1 = 60  # if low then more weak edges will be found so weak circles returned (default 60)
    param2 = 30  # if low then more circles will be returned by HoughCircles (default 30)

    # apply OpenCV HoughCircle algorithm
    circles = cv.HoughCircles(blur_image, cv.HOUGH_GRADIENT, dp, min_dist,
                              param1=param1, param2=param2,
                              minRadius=min_r, maxRadius=max_r)

    # get centres and radius into np.array
    dot_location = np.array(circles)
    dot_location = np.squeeze(dot_location, axis=0)  # remove redundant dimension

    # initialise final image
    final_image = initial_image

    # remove non-red dots:
    blue_thresh = 100  # must be less than this to be red dot (default 80)
    green_thresh = 80  # must be less than this to be red dot (default 80)
    red_thresh = 100  # must be greater than this to be red dot (default 100)

    # add column to show think it is false positive
    z1 = np.zeros((np.shape(dot_location)[0], 1))
    dot_location = np.concatenate((dot_location, z1), axis=1)

    # flag false pos by checking colour
    for i in range(np.shape(dot_location)[0]):
        pix_check_x = int(dot_location[(i, 0)])  # grab x coord
        pix_check_y = int(dot_location[(i, 1)])  # grab y coord

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
        if blue > blue_thresh or green > green_thresh or red < red_thresh:
            dot_location[i, 3] = 1

    # remove all flagged presumed false positives and remove added flag column
    dot_location = np.delete(dot_location, np.where(dot_location[:, 3] == 1)[0], 0)
    dot_location = np.delete(dot_location, np.s_[-1:], axis=1)

    # write colour as text
    font = cv.FONT_HERSHEY_SIMPLEX  # set font
    for i in range(np.shape(dot_location)[0]):
        pix_check_x = int(dot_location[(i, 0)])  # grab x coord
        pix_check_y = int(dot_location[(i, 1)])  # grab y coord

        blue = initial_image[pix_check_y, pix_check_x, 0]
        green = initial_image[pix_check_y, pix_check_x, 1]
        red = initial_image[pix_check_y, pix_check_x, 2]

        cv.putText(final_image, str(blue),
                   (int(dot_location[i, 0]), int(dot_location[i, 1]) - 50),
                   font, 0.7, (255, 0, 0), 2)
        cv.putText(final_image, str(green),
                   (int(dot_location[i, 0]) + 50, int(dot_location[i, 1]) - 50),
                   font, 0.7, (0, 255, 0), 2)
        cv.putText(final_image, str(red),
                   (int(dot_location[i, 0] + 100), int(dot_location[i, 1]) - 50),
                   font, 0.7, (0, 0, 255), 2)

    # draw the detected circles
    if circles is not None:
        # removes decimals
        circles_draw = np.uint16(np.around(dot_location))
        # print('circles drawn:', circles_draw)
        for i in circles_draw:
            center = (i[0], i[1])
            # circle center
            cv.circle(final_image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            # draws circles (r,b,g) colour
            cv.circle(final_image, center, radius, (0, 255, 0), 10)

    # save image as filename.jpeg
    cv.imwrite(str('images_processed/TestSet/' + picture + '/' + '1red_dot_location.jpg'), final_image)

    # check if 2 dots located
    if np.shape(dot_location)[0] != 2:
        print('ERROR: EXACTLY 2 RED DOTS FOUND')

    return dot_location


# function to return distance between 2 points
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


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
    fneg_weight = 3
    e_total = (e_loc + (no_fp * fpos_weight) + (no_fn * fneg_weight))  # total error equation

    return no_fp, no_fn, no_correct, e_loc, e_total


# function to calculate the number of false positives and negatives
def false_pos_neg(estimate, ground_truth):
    """
    Function returns:
     - an array of the estimates with flag 1 if they are false positives (wrongly circled).
     - an array of the ground truths with flag 1 if they are false negatives (not found).
    """

    # threshold pixel distance for being false pos and false neg
    threshold = 5  # mm

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


def draw_error(estimate, ground_truth, pix_locations, picture):
    # select centres only for pixel location estimates
    screw_centres_found = estimate[:, :2]

    filename = str('images_taken/ToTest/' + picture + '.jpg')

    # loads an image
    image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # output false pos and false neg locations
    false_pos, false_neg = false_pos_neg(screw_centres_found, ground_truth)

    # draw false positives (wrongly labelled screws) as yellow circle
    false_pos_loc = np.delete(pix_locations, np.where(false_pos[:, 2] == 0)[0], 0)  # find false pos locations & radii
    no_of_fp = np.shape(false_pos_loc)[0]
    for i in range(no_of_fp):
        image = cv.circle(image, np.uint16(false_pos_loc[i, (0, 1)]), radius=np.uint16(false_pos_loc[i, 2]),
                          color=(0, 255, 255), thickness=4)

    # draw false negatives (missed screws) as red circle
    #false_neg_loc = np.delete(false_neg, np.where(false_neg[:, 2] == 0)[0], 0)  # find false neg locations
    #no_of_fn = np.shape(false_neg_loc)[0]
    #for i in range(no_of_fn):
    #    image = cv.circle(image, np.uint16(false_neg_loc[i, (0, 1)]), radius=23,
    #                      color=(0, 0, 255), thickness=4)

    # draw correctly found estimates as pink circle
    correct_found_loc = np.delete(pix_locations, np.where(false_pos[:, 2] == 1)[0], 0)  # find locations & radii
    no_of_correct = np.shape(correct_found_loc)[0]
    for i in range(no_of_correct):
        image = cv.circle(image, np.uint16(correct_found_loc[i, (0, 1)]), radius=np.uint16(correct_found_loc[i, 2]),
                          color=(255, 0, 255), thickness=3)

    # draw ground truths as green dot
    #found_gt_loc = false_neg
    #no_of_gt = np.shape(found_gt_loc)[0]
    #for i in range(no_of_gt):
    #    image = cv.circle(image, np.uint16(found_gt_loc[i, (0, 1)]), radius=7,
    #                      color=(0, 255, 0), thickness=-1)

    # color key
    cv.putText(image, text="KEY", org=(1400, 1450), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=4, color=(0, 0, 0), thickness=3)
    cv.putText(image, text=str(no_of_fp) + " False positive screws", org=(1400, 1530), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=3, color=(0, 255, 255), thickness=2)
    #cv.putText(image, text=str(no_of_fn) + " Screws not found", org=(1400, 1600), fontFace=cv.FONT_HERSHEY_DUPLEX,
    #           fontScale=3, color=(0, 0, 255), thickness=2)
    cv.putText(image, text=str(no_of_correct) + " Screws correctly found", org=(1400, 1670), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=3, color=(255, 0, 255), thickness=2)
    #cv.putText(image, text=str(no_of_gt) + " Ground truths", org=(1400, 1740), fontFace=cv.FONT_HERSHEY_DUPLEX,
    #           fontScale=3, color=(0, 255, 0), thickness=2)

    # resize and show image
    #resized_image = resize_to_fit_screen.resize(image, 1000)
    #cv.imshow("screw error", resized_image)

    # save image as filename.jpeg

    save_image = str('images_processed/TestSet/' + picture + '/' + 'mm_error.jpg')
    cv.imwrite(save_image, image)

    #cv.waitKey(0)  # wait till user exits or presses q

    return

if __name__ == "__main__":
    # set picture
    picture = 'phone_picTest8'

    pix_to_mm, tl_corner_pix = calibrate_camera(picture=picture, mm_dist=80)

    # adjust tl_corner_pix to screw
    tl_corner_pix = np.loadtxt(str('images_processed/TestSet/' + picture + '/' + 'origin_pix.txt'), delimiter=",")

    screw_locations = mm_screw_location(pix_to_mm, tl_corner_pix, picture=picture)
    pix_locations = pixel_screw_location(picture=picture)
    screw_centres = screw_locations[:, :2]

   # ground_truths = np.loadtxt(str('images_processed/TestSet/' + picture + '/' + 'mm_screw_ground_truths .txt'),
   #                            delimiter=",")

  #  no_fp, no_fn, no_correct, e_loc, e_total = total_error(screw_centres, ground_truths)
  #  draw_error(screw_centres, ground_truths, pix_locations, picture)

    # print outputs
  #  print('There are', no_fp, 'screws falsely labelled,', no_fn, 'screws that were missed and',
   #       no_correct, 'correctly found.')
   # print('The location error of correctly found screws is:', e_loc, 'mm.')
  #  print('')
   # print('total error:', e_total)
