#!/usr/bin/env python3

"""

Nathan Wooster
Feb 2022
Script for chess board camera calibration.

"""

import cv2 as cv
from resize import resize


def calibrate_camera(columns, rows):
    """
        `columns` and `rows` should be the number of inside corners in the
        chessboard's columns and rows.
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

        # pixel checks
        corner_location = corners_sub_pix  # order goes up from bottom of column1, then next column starts bottom
        print(corner_location)
        pixel_size = img.shape[:2]
        print('x:', pixel_size[1], 'y:', pixel_size[0])  # longer x creates rectangle
        cv.circle(img, (407, 1702), 50, (0, 0, 255))  # draw a circle somewhere
        cv.circle(img, (408, 1648), 50, (0, 255, 0))  # draw a circle somewhere
        cv.circle(img, (462, 1703), 50, (255, 0, 0))  # draw a circle somewhere

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


if __name__ == "__main__":
    calibrate_camera(7, 7)
