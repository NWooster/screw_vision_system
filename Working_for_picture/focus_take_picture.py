#!/usr/bin/env python3

'''

Nathan Wooster
Feb 2022
Script for accessing camera.

'''
import sys
import cv2 as cv
import time
import numpy as np


def take_picture(camera):
    # define camera (width pixels must be 0.75 of height)
    pixel_w = 1944  # default 1944
    pixel_h = pixel_w/0.75  # default 2592 (ratio 0.75)
    cap = cv.VideoCapture(camera)  # video capture source camera (Here external webcam)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, pixel_w)  # set desired pixel width (default 1944)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, pixel_h)  # set desired pixel height (default 2592)
    print('camera connected')

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Check if webcam connected properly")  # print and error message
    else:
        start_time = time.time()
        autofocus_duration = 4
        print('wait ' + str(autofocus_duration) + ' secs for autofocus')
        while (int(time.time() - start_time) < autofocus_duration):
            ret, frame = cap.read()
            if ret == 1:
                pass
            else:
                print('error with frame')

        print('autofocus complete')

        ret, frame = cap.read()

        pixel_size = frame.shape[:2]
        no_of_pixels = (frame.shape[0] * frame.shape[1])/1000000
        print('pixel size:', pixel_size)
        print('no. of pixels:', no_of_pixels, 'MP')

        cv.imwrite('image_from_new_cam' + '.jpg', frame)

    # After the loop release the cap object
    cap.release()
    # out.release()
    # Destroy all the windows
    cv.destroyAllWindows()

    #    if cv.waitKey(1) & 0xFF == ord('q'):
    #       cap.release()  # release the camera object
    #      cv.destroyAllWindows()  # close all other windows

    return 0


if __name__ == "__main__":
    camera_no = 1
    take_picture(camera_no)
