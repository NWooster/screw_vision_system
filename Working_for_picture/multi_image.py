#!/usr/bin/env python3

'''

Nathan Wooster
23/12/21
The main python script to run screw image detection system

'''
import sys
import cv2 as cv
import numpy as np

from resize import resize


# loads image, pre-process it, apply hough circle detection
def main(argv):
    '''main function called to run the vision algorithm'''

    # labels where image is
    image_file = 'pics/pic2.jpg'
    filename = argv[0] if len(argv) > 0 else image_file

    # loads an image and calls it 'initial_image'
    initial_image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # check if image is loaded fine
    if initial_image is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + image_file + '] \n')
        return -1

    # convert image to grayscale from BGR and new image called 'gray'
    gray = cv.cvtColor(initial_image, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray image', gray)

    # adds medium blur to image to reduce noise (avoids false circle detection)
    blur_image = cv.medianBlur(gray, 5)
    blur_resized = resize(blur_image, 600)
    cv.imshow('Blur image', blur_resized)

    # numpy array .shape[0] outputs the number of elements in dimension 1 of the array (number of pixel rows)
    rows = blur_image.shape[0]
    # print(rows)
    # print(blur_image)

    '''
    Hough circle algorithm arguments:

    gray: Input image (grayscale).
    circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
    HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
    dp = 1: The inverse ratio of resolution.
    min_dist = gray.rows/8: Minimum distance between detected centers.
    param_1 = 100: Upper threshold for the internal Canny edge detector.
    param_2 = 30*: Threshold for center detection.
    min_radius = 1: Minimum radius to be detected. If unknown, put zero as default.
    max_radius = 30: Maximum radius to be detected. If unknown, put zero as default.
    '''
    circles = cv.HoughCircles(blur_image, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=5, maxRadius=60)
    # circles_og = circles
    # print('circles_og:', circles_og)

    # initialise final image
    final_image = initial_image

    # draw the detected circles
    if circles is not None:
        # removes decimals
        circles = np.uint16(np.around(circles))
        # print('circles:', circles)
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(final_image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            # draws circles (r,b,g) colour
            cv.circle(final_image, center, radius, (255, 0, 255), 3)

    # call imported resize image function specify required width
    resized_image = resize(final_image, 600)
    # show resized image
    cv.imshow("detected circles", resized_image)
    # wait for user to press exit
    cv.waitKey(0)

    return 0


# sys.argv[1:] is the user input (empty array -  good practise)

if __name__ == "__main__":
    # print(sys.argv[1:])
    main(sys.argv[1:])
