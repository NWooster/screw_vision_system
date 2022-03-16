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


def calibrate_camera(image_location='images_taken/1latest_image_from_camera', columns=7, rows=7, width=25, height=25):
    """
        `columns` and `rows` are the number of INSIDE corners in the
        chessboard's columns and rows.
        'width' is the length in mm of the side of the chessboard.
        'height' is the length in mm of the side of the chessboard.
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

        # select 4 corners (B,G,R)
        # sometimes changes so just edit name based off colour
        br_corner = corner_location[columns * rows - rows, :]  # extract top left corner
        br_corner_rnd = np.rint(br_corner)  # round pixel float to nearest integer
        cv.circle(img, (int(br_corner_rnd[0]), int(br_corner_rnd[1])), 50, (0, 0, 0))  # draw circle at location

        bl_corner = corner_location[0, :]  # extract top right corner
        bl_corner_rnd = np.rint(bl_corner)  # round pixel float to nearest integer
        cv.circle(img, (int(bl_corner_rnd[0]), int(bl_corner_rnd[1])), 50, (0, 0, 255))  # draw circle at location

        tl_corner = corner_location[columns - 1, :]  # extract bottom right corner
        tl_corner_rnd = np.rint(tl_corner)  # round pixel float to nearest integer
        cv.circle(img, (int(tl_corner_rnd[0]), int(tl_corner_rnd[1])), 50, (0, 255, 0))  # draw circle at location

        tr_corner = corner_location[columns * rows - 1, :]  # extract bottom left corner
        tr_corner_rnd = np.rint(tr_corner)  # round pixel float to nearest integer
        cv.circle(img, (int(tr_corner_rnd[0]), int(tr_corner_rnd[1])), 50, (255, 0, 0))  # draw circle at location

        # calc pixel distance between corners
        pix_width1 = distance(tl_corner_rnd[0], tl_corner_rnd[1], tr_corner_rnd[0], tr_corner_rnd[1])  # top side
        pix_width2 = distance(bl_corner_rnd[0], bl_corner_rnd[1], br_corner_rnd[0], br_corner_rnd[1])  # bottom side
        pix_height1 = distance(tl_corner_rnd[0], tl_corner_rnd[1], bl_corner_rnd[0], bl_corner_rnd[1])  # left side
        pix_height2 = distance(tr_corner_rnd[0], tr_corner_rnd[1], br_corner_rnd[0], br_corner_rnd[1])  # right side

        # calc error range from two different possible corners
        pix_width_error = abs(pix_width1 - pix_width2)
        pix_height_error = abs(pix_height1 - pix_height2)
        # print('width error in pixels:', pix_width_error)
        # print('height error in pixels:', pix_height_error)

        # use average to calc lengths of whole board
        ave_pix_width = (pix_width1 + pix_width2) / 2  # width of inside board
        ave_pix_height = (pix_height1 + pix_height2) / 2  # height of inside board
        pix_all_width = (ave_pix_width / rows - 1) * (rows + 1)  # width of whole board
        pix_all_height = (ave_pix_height / columns - 1) * (columns + 1)  # height of whole board

        # calculate pixel to mm ratio
        mm_width = width  # mm
        mm_height = height  # mm
        ratio1 = mm_width / pix_all_width  # 1 pixel is this many mm
        ratio2 = mm_height / pix_all_height  # 1 pixel is this many mm
        ratio_error = abs(ratio1 - ratio2)
        pix_mm_ratio = (ratio1 + ratio2) / 2  # 1 pixel is this many mm
        # print('1 pixel is ' + str(pix_mm_ratio) + 'mm and 1mm is ' + str(1 / pix_mm_ratio) + ' many pixels')
        # print('possible error in mm per pixel is ' + str(ratio_error) + ' so max cumulative error is ' +
        #       str(max(pixel_size[:]) * ratio_error) + 'mm')

        # draw corners onto image
        cv.drawChessboardCorners(img, (columns, rows), corners_sub_pix, ret)
        resized_image = resize(img, 800)  # resize image to fit screen
        # save non-resized image
        cv.imwrite('images_processed/1chequered_cal' + '.jpg', img)

        # return top left corner coordinates in pixels
        tl_corner_pix = np.array(tl_corner)

        # print()
        # print('press "q" to exit')
        # show image until user presses 'q'
        # while True:
        #    cv.imshow('found corners', resized_image)
        #    # press 'q' button to exit image
        #    if cv.waitKey(1) & 0xFF == ord('q'):
        #        break

    return pix_mm_ratio, tl_corner_pix, ratio_error


# function to return distance between 2 points
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


if __name__ == "__main__":
    output = calibrate_camera(image_location='images_taken/with_25white_square.jpg')
    print(output)
