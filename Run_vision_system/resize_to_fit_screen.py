#!/usr/bin/env python3

"""
resize.py
Nathan Wooster
Jan 2022
This script contains the resize function to
fit an image to the screen without altering
the aspect ratio.
"""

import cv2 as cv


def resize(image, req_width):

    """
    Function to return resized
    image with correct aspect ratio.
    """

    # find current pixel height and width
    height, width = image.shape[:2]
    # calc new_height with correct aspect ratio
    ratio = width / height
    new_height = req_width / ratio
    # check if pixel height is an integer
    integer_check = new_height.is_integer()

    # if pixel height is not an integer return error message
    if not integer_check:
        print('error! image ratio of:', ratio, 'and given width of:', req_width, 'does not output an integer for pixel '
                                                                                 'height:', new_height)
        return 0

    else:
        # convert float to integer for cv function
        new_height = int(float(new_height))
        # return resized image
        resized_im = cv.resize(image, (req_width, new_height))  # Resize image
        return resized_im
