#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import resize_to_fit_screen

filename = 'images_taken/straight_edge2.jpg'

image = cv.imread(filename)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
r_image = resize_to_fit_screen.resize(image, 1000)
#cv.imshow("image", r_image)

# crop image
h = 1844
w = 2592
image = image[0:0+h, 0:0+w]


edge = cv.Canny(image, 60, 180)
r_edge = resize_to_fit_screen.resize(edge, 1000)
#cv.imshow("edge", r_edge)

contours, h = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv.contourArea, reverse=True)
cv.drawContours(image, contours[0], -1, (0, 0, 255), thickness=5)
r_image = resize_to_fit_screen.resize(image, 1000)
cv.imshow("image", r_image)




cv.waitKey(0)