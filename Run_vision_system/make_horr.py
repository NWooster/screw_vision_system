#!/usr/bin/env python3

"""

Nathan Wooster
Feb 2022
Script for checking two holes are horizontal.

"""

import cv2 as cv
import numpy as np

# custom imports
import take_picture


def horizontal_check(image_location='images_taken/1latest_image_from_camera.jpg'):
    # labels where image is
    filename = image_location

    # loads an image and calls it 'initial_image'
    initial_image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # check if image is loaded fine
    if initial_image is None:
        print('Error opening image!')
        return -1

    gray = cv.cvtColor(initial_image, cv.COLOR_BGR2GRAY)
    blur_image = cv.medianBlur(gray, 5)

    # parameters for Hough Circle algorithm
    dp = 1  # high dp means low matrix resolution so takes circles that do not have clear boundary (default 1)
    min_r = 40  # min pixel radius of screw (default 18)
    max_r = 80  # max pixel radius of screw (default 30)
    min_dist = int(min_r * 2)  # min distance between two screws
    param1 = 60  # if low then more weak edges will be found so weak circles returned (default 60)
    param2 = 30  # if low then more circles will be returned by HoughCircles (default 30)

    # apply OpenCV HoughCircle algorithm
    circles = cv.HoughCircles(blur_image, cv.HOUGH_GRADIENT, dp, min_dist,
                              param1=param1, param2=param2,
                              minRadius=min_r, maxRadius=max_r)

    # get centres and radius into np.array
    hole_locations = np.array(circles)
    hole_locations = np.squeeze(hole_locations, axis=0)  # remove redundant dimension

    # initialise final image
    final_image = initial_image

    # draw the detected circles
    if circles is not None:
        # removes decimals
        circles_draw = np.uint16(np.around(hole_locations))
        # print('circles drawn:', circles_draw)
        for i in circles_draw:
            center = (i[0], i[1])
            # circle center
            cv.circle(final_image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            # draws circles (r,b,g) colour
            cv.circle(final_image, center, radius, (255, 0, 255), 3)

    # check which one is top left, top right, bottom left, bottom right
    hole_locations = hole_locations[hole_locations[:, 1].argsort()]  # re-order based on y coordinate

    if hole_locations[0, 0] < hole_locations[1, 0]:
        tl = hole_locations[0, :]
        tr = hole_locations[1, :]
    else:
        tl = hole_locations[1, :]
        tr = hole_locations[0, :]

    if hole_locations[2, 0] < hole_locations[3, 0]:
        bl = hole_locations[2, :]
        br = hole_locations[3, :]
    else:
        bl = hole_locations[3, :]
        br = hole_locations[2, :]

    # check assigned corners correctly
    if tl[1] < bl[1] and tl[0] < tr[0] and tr[1] < br[1] and br[0] > bl[0]:
        # everything correct
        pass
    else:
        print('error corners not in the right location')

    # calc difference
    pixel_x_diff1 = tl[0] - bl[0]
    pixel_x_diff2 = tr[0] - br[0]
    ave_x_diff = (pixel_x_diff1 + pixel_x_diff2)/2

    pixel_y_diff1 = tl[1] - tr[1]
    pixel_y_diff2 = bl[1] - br[1]
    ave_y_diff = (pixel_y_diff1 + pixel_y_diff2)/2

    # print instructions
    # x
    if tl[0] > bl[0] and tr[0] > br[0]:
        print('rotate camera clockwise to account for x: ' + str(ave_x_diff) + 'pixels')
    elif tl[0] < bl[0] and tr[0] < br[0]:
        print('rotate camera anticlockwise to account for x: ' + str(ave_x_diff) + 'pixels')
    else:
        print('x difference do not agree, left:' + str(pixel_x_diff1) + '  right:' + str(pixel_x_diff2))
        if abs(pixel_x_diff1) > abs(pixel_x_diff2):
            if tl[0] > bl[0]:
                print('however probably rotate clockwise')
            else:
                print('however probably rotate anticlockwise')
        if abs(pixel_x_diff2) > abs(pixel_x_diff1):
            if tr[0] > br[0]:
                print('however probably rotate clockwise')
            else:
                print('however probably rotate anticlockwise')

    # y
    if tl[1] > tr[1] and bl[1] > br[1]:
        print('rotate camera anticlockwise to account for y: ' + str(ave_y_diff) + 'pixels')
    elif tl[1] < tr[1] and bl[1] < br[1]:
        print('rotate camera clockwise to account for y: ' + str(ave_y_diff) + 'pixels')
    else:
        print('y difference do not agree, top:' + str(pixel_y_diff1) + 'bottom:' + str(pixel_y_diff2))
        if abs(pixel_y_diff1) > abs(pixel_y_diff2):
            if tl[1] > tr[1]:
                print('however probably rotate anticlockwise')
            else:
                print('however probably rotate clockwise')
        if abs(pixel_y_diff2) > abs(pixel_y_diff1):
            if bl[1] > br[1]:
                print('however probably rotate anticlockwise')
            else:
                print('however probably rotate clockwise')

    # draw circle at top left BLACK
    cv.circle(final_image, (int(tl[0]), int(tl[1])), 100, (0, 0, 0))  # tl is BLACK

    # draw circle at top right RED
    cv.circle(final_image, (int(tr[0]), int(tr[1])), 100, (0, 0, 255))

    # draw circle at bottom left GREEN
    cv.circle(final_image, (int(bl[0]), int(bl[1])), 100, (0, 255, 0))

    # draw circle at bottom right LIGHT BLUE
    cv.circle(final_image, (int(br[0]), int(br[1])), 100, (255, 255, 0))

    # save image as filename.jpeg
    cv.imwrite('images_processed/horizontal_holes' + '.jpg', final_image)


if __name__ == "__main__":
    nathan_webcam = 1
    take_picture.take_picture(nathan_webcam, 10)  # take image from webcam (camera 1) with specified autofocus time
    horizontal_check()
