#!/usr/bin/env python3

'''

Nathan Wooster
Jan 2022
The main python script to run screw image detection system

'''
import sys
import cv2 as cv
import numpy as np

# custom import
from resize import resize


# loads image, pre-process it, apply hough circle detection
def main(argv):
    """main function called to run the vision algorithm"""

    # labels where image is
    #image_file = 'auto_save_images/1latest_image_from_camera.jpg'  # default pic8
    image_file = 'pictures_from_rig/149_square.jpg'
    filename = argv[0] if len(argv) > 0 else image_file  # allows user to input a file name

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
    dp = 1  # high dp means low matrix resolution so takes circles that do not have clear boundary (default 1)
    min_r = 12  # min pixel radius of screw
    max_r = 30  # max pixel radius of screw
    min_dist = int(min_r * 4)  # min distance between two screws
    param1 = 90  # if low then more weak edges will be found so weak circles returned (default 100)
    param2 = 30  # if low then more circles will be returned by HoughCircles (default 30)

    # apply OpenCV HoughCircle algorithm
    circles = cv.HoughCircles(blur_image, cv.HOUGH_GRADIENT, dp, min_dist,
                              param1=param1, param2=param2,
                              minRadius=min_r, maxRadius=max_r)
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

    # call imported resize image function specify required width (default 600)
    ##resized_image = resize(final_image, 600)
    # show resized image
    ##cv.imshow("detected screws", resized_image)

    # save image as filename.jpeg
    #cv.imwrite('auto_save_images/2camera_screws_detected' + '.jpg', final_image)
    cv.imwrite('pictures_from_rig/post_process/1screw_output' + '.jpg', final_image)

    # wait for user to press exit
    cv.waitKey(0)

    return 0


# sys.argv[1:] is the user input (empty array -  good practise)

if __name__ == "__main__":
    # print(sys.argv[1:])
    main(sys.argv[1:])
