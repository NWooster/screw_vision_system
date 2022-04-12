#!/usr/bin/env python3

"""

Nathan Wooster
Feb 2022
Script for chess board camera calibration.

"""

import cv2 as cv
import numpy as np
import math
from resize_to_fit_screen import resize


def calibrate_camera(image_location='images_taken/1latest_image_from_camera', mm_dist=80,
                     columns=7, rows=7):
    """
        `columns` and `rows` are the number of INSIDE corners in the
        chessboard's columns and rows.
        'mm_dist' is the distance in mm between each red dot.
    """

    # load image
    filename = image_location  # image location
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
        red_dot_pix = find_red_dot(image_location=filename)

        # find red dot pix distance from each other
        red_dot_dist = distance(red_dot_pix[0, 0], red_dot_pix[0, 1], red_dot_pix[1, 0], red_dot_pix[1, 1])

        # calc mm to pixel ratio
        pix_mm_ratio = mm_dist/red_dot_dist

        # draw corners onto image
        cv.drawChessboardCorners(img, (columns, rows), corners_sub_pix, ret)
        resized_image = resize(img, 800)  # resize image to fit screen
        # save non-resized image
        cv.imwrite('images_processed/1chequered_cal' + '.jpg', img)

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


def find_red_dot(image_location='images_taken/1latest_image_from_camera.jpg'):
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
                   (int(dot_location[i, 0])+50, int(dot_location[i, 1]) - 50),
                   font, 0.7, (0, 255, 0), 2)
        cv.putText(final_image, str(red),
                   (int(dot_location[i, 0]+100), int(dot_location[i, 1]) - 50),
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
    cv.imwrite('images_processed/1red_dot_location' + '.jpg', final_image)

    # check if 2 dots located
    if np.shape(dot_location)[0] != 2:
        print('ERROR: EXACTLY 2 RED DOTS FOUND')

    return dot_location


# function to return distance between 2 points
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


if __name__ == "__main__":
    output = calibrate_camera(image_location='images_taken/1latest_image_from_camera.jpg')
    print(output)
