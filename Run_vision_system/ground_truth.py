#!/usr/bin/env python3

import cv2 as cv
import numpy as np

# This variable we use to store the pixel location
pixel_loc = []


# click event function
def click_event(event, x, y, flags, param):
    # re-adjust x for crop
    x_adjust = x + x_lower

    # shows pixel location in blue
    if event == cv.EVENT_LBUTTONDOWN:
        print(x_adjust, ",", y)
        pixel_loc.append([x_adjust, y])
        font = cv.FONT_HERSHEY_SIMPLEX
        str_xy = str(x_adjust) + ", " + str(y)
        cv.putText(img, str_xy, (x, y), font, 0.5, (255, 255, 0), 2)
        cv.imshow("image", img)

    # shows the colour composition of that pixel in yellow
    if event == cv.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv.FONT_HERSHEY_SIMPLEX
        str_bgr = str(blue) + ", " + str(green) + ", " + str(red)
        cv.putText(img, str_bgr, (x, y), font, 0.5, (0, 255, 255), 2)
        cv.imshow("image", img)


# specify what side of image to show
#side_of_image = "left"
side_of_image = "right"

# image path
filename = 'images_taken/1latest_image_from_camera.jpg'
img = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

# output image size
pixel_size = img.shape[:2]
# print(pixel_size)

if side_of_image == "right":

    # these do not change
    y_upper = pixel_size[0] - 500
    y_lower = 0
    x_upper = 2592

    x_lower = 1920  # IMPORTANT crop place

    # crop image
    img = img[y_lower:y_upper, x_lower:x_upper]

    # show image
    cv.imshow("image", img)
    # calling the mouse click event
    cv.setMouseCallback("image", click_event)

elif side_of_image == "left":

    x_lower = 0  # for function

    # show image
    cv.imshow("image", img)
    # calling the mouse click event
    cv.setMouseCallback("image", click_event)

else:
    print('error: specify side of image to show')

cv.waitKey(0)  # wait till user exits or presses q

# save the pixel locations to .txt file
screw_ground_truths = np.array(pixel_loc)
print(screw_ground_truths)
np.savetxt(side_of_image + "_side_screw_ground_truths.txt", screw_ground_truths, fmt='%g', delimiter=",")

cv.destroyAllWindows()
