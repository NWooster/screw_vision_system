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
        ``columns`` and ``rows`` should be the number of inside corners in the
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
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub_pix = cv.cornerSubPix(gray, found_corners, (5, 5), (-1, -1), criteria)

        # draw corners onto image
        cv.drawChessboardCorners(img, (columns, rows), corners_sub_pix, ret)
        resized_image = resize(img, 800)  # resize image to fit screen

        # show image until user presses 'q'
        while True:
            cv.imshow('found corners', resized_image)
            # press 'q' button to exit image
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    return found_corners


if __name__ == "__main__":
    calibrate_camera(7, 7)
