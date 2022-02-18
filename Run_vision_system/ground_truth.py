#!/usr/bin/env python3

import cv2 as cv
import numpy as np

from resize_to_fit_screen import resize

# This will display all the available mouse click events
events = [i for i in dir(cv) if 'EVENT' in i]
print(events)

# This variable we use to store the pixel location
refPt = []


# click event function
def click_event(event, x, y, flags, param):

    # shows pixel location in blue
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ",", y)
        refPt.append([x, y])
        font = cv.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ", " + str(y)
        cv.putText(img, strXY, (x, y), font, 0.5, (255, 255, 0), 2)
        cv.imshow("image", img)

    # shows the colour composition of that pixel in yellow
    if event == cv.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ", " + str(green) + ", " + str(red)
        cv.putText(img, strBGR, (x, y), font, 0.5, (0, 255, 255), 2)
        cv.imshow("image", img)


# Here, you need to change the image name and it's path according to your directory
filename = 'images_processed/1screw_output.jpg'
img = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

# resized_image = resize(img, 600)

cv.imshow("image", img)

# calling the mouse click event
cv.setMouseCallback("image", click_event)

cv.waitKey(0)
cv.destroyAllWindows()
