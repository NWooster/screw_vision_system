#!/usr/bin/env python3

'''

Nathan Wooster
Jan 2022
Script for edge detection

'''
from __future__ import print_function
import sys
import cv2 as cv
import argparse
import numpy as np

# custom import
from resize import resize


def edge_detection(image):
    """edge detection script"""

    # labels where image is
    filename = image  # default pic8

    # loads an image and calls it 'initial_image'
    initial_image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # check if image is loaded fine
    if initial_image is None:
        print('Error opening image!')
        return -1

    # convert image to grayscale from BGR and new image called 'gray'
    gray = cv.cvtColor(initial_image, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray image', gray)

    # adds blur to image to reduce noise
    blur_image = cv.blur(gray, (5, 5))  # blur in (x, y) larger number more blur
    # blur_resized = resize(blur_image, 600)
    # cv.imshow('Blur image', blur_resized)

    # numpy array .shape[0] outputs the number of elements in dimension 1 of the array (number of pixel rows)
    # rows = blur_image.shape[0]

    # initialise final image
    final_image = initial_image

    # parameters for Canny Edge Detection algorithm
    low_threshold = 0
    ratio = 3
    kernel_size = 3

    detected_edges = cv.Canny(blur_image, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    final_image = final_image * (mask[:, :, None].astype(final_image.dtype))

    # call imported resize image function specify required width
    resized_image = resize(final_image, 600)
    # show resized image
    cv.imshow("detected edges", resized_image)
    # wait for user to press exit
    cv.waitKey(0)

    return 0


# sys.argv[1:] is the user input (empty array -  good practise)

if __name__ == "__main__":
    #edge_detection('pics/pic8.jpg')
    cv.createTrackbar('Min Threshold:', "detected edges", 0, 100, edge_detection('pics/pic8.jpg'))
    # main(sys.argv[1:])
