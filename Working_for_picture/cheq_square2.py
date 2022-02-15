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

    filename = 'pictures_from_rig/149_square.jpg'

    img = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    ret, found_corners = cv.findChessboardCorners(img, (columns, rows),
                                                  flags=cv.CALIB_CB_FAST_CHECK)

    # corners not found
    if ret != 1:
        print('chess board corners not found!')

    # corners found
    else:
        print('corners found')

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        #found_corners2 = cv.cornerSubPix(img, found_corners, (5, 5), (-1, -1), criteria)

        #cv.drawChessboardCorners(img, (columns, rows), found_corners2, ret)
        #resized_image2 = resize(img, 800)  # resize image to fit screen

        # draw corners onto image
        cv.drawChessboardCorners(img, (columns, rows), found_corners, ret)
        resized_image = resize(img, 800)  # resize image to fit screen

        # show image until user presses 'q'
        while True:
            cv.imshow('found corners', resized_image)
            #cv.imshow('found corners2', resized_image2)

            # press 'q' button to exit image
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    return found_corners


if __name__ == "__main__":
    calibrate_camera(7, 7)
