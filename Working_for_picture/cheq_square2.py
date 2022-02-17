#!/usr/bin/env python3

"""

Nathan Wooster
Feb 2022
Script for chess board camera calibration.

"""

import cv2 as cv
import numpy as np
import math
from resize import resize


def calibrate_camera(columns, rows, width, height):
    """
        `columns` and `rows` are the number of inside corners in the
        chessboard's columns and rows.
        'width' is the length in mm of the side of the chessboard.
        'height' is the length in mm of the side of the chessboard.
        """

    # load image
    filename = 'pictures_from_rig/with_25_square.jpg'  # default 149_square.jpg
    img = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, found_corners = cv.findChessboardCorners(gray, (columns, rows),
                                                  flags=cv.CALIB_CB_FAST_CHECK)

    # corners not found
    if ret != 1:
        print('chess board corners not found!')

    # corners found
    else:
        print('corners found')

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
        corner_location = np.array(corners_sub_pix)  # goes up from bottom of column1, then next column starts bottom
        corner_location = np.squeeze(corner_location, axis=1)  # remove redundant array dimension
        # pixel_size = img.shape[:2]
        # print('x:', pixel_size[1], 'y:', pixel_size[0])  # longer x creates rectangle
        # cv.circle(img, (407, 1702), 50, (0, 0, 255))  # draw a circle somewhere

        # select 4 corners
        bl_corner = corner_location[0, :]  # extract bottom left corner
        bl_corner_rnd = np.rint(bl_corner)  # round pixel float to nearest integer
        cv.circle(img, (int(bl_corner_rnd[0]), int(bl_corner_rnd[1])), 50, (0, 0, 255))  # draw circle at location

        tl_corner = corner_location[6, :]  # extract top left corner
        tl_corner_rnd = np.rint(tl_corner)  # round pixel float to nearest integer
        cv.circle(img, (int(tl_corner_rnd[0]), int(tl_corner_rnd[1])), 50, (0, 255, 0))  # draw circle at location

        br_corner = corner_location[42, :]  # extract bottom right corner
        br_corner_rnd = np.rint(br_corner)  # round pixel float to nearest integer
        cv.circle(img, (int(br_corner_rnd[0]), int(br_corner_rnd[1])), 50, (255, 0, 0))  # draw circle at location

        tr_corner = corner_location[48, :]  # extract top right corner
        tr_corner_rnd = np.rint(tr_corner)  # round pixel float to nearest integer
        cv.circle(img, (int(tr_corner_rnd[0]), int(tr_corner_rnd[1])), 50, (255, 255, 0))  # draw circle at location

        # calc pixel distance
        pix_width1 = distance(tl_corner_rnd[0], tl_corner_rnd[1], tr_corner_rnd[0], tr_corner_rnd[1])
        print(pix_width1)
        pix_width2 = distance(bl_corner_rnd[0], bl_corner_rnd[1], br_corner_rnd[0], br_corner_rnd[1])
        print(pix_width2)
        pix_height1 = distance(tl_corner_rnd[0], tl_corner_rnd[1], bl_corner_rnd[0], bl_corner_rnd[1])
        print(pix_height1)
        pix_height2 = distance(tr_corner_rnd[0], tr_corner_rnd[1], br_corner_rnd[0], br_corner_rnd[1])
        print(pix_height2)

        ##ADD extra 2 squares!


        mm_width = width  # mm
        mm_height = height  # mm

        # draw corners onto image
        cv.drawChessboardCorners(img, (columns, rows), corners_sub_pix, ret)
        resized_image = resize(img, 800)  # resize image to fit screen
        # save non-resized image
        cv.imwrite('pictures_from_rig/post_process/1chequered_cal' + '.jpg', img)
        # show image until user presses 'q'
        while True:
            cv.imshow('found corners', resized_image)
            # press 'q' button to exit image
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    return found_corners


# function to return distance between 2 points
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


if __name__ == "__main__":
    calibrate_camera(7, 7, 25, 25)
