#!/usr/bin/env python3

"""

Nathan Wooster
Feb 2022
Script for accessing camera.

"""


import cv2 as cv
import time


def take_picture(camera):

    """Take picture function including time to autofocus camera"""

    # define camera (width pixels must be 0.75 of height)
    pixel_w = 1944  # default 1944
    pixel_h = pixel_w/0.75  # default 2592 (ratio 0.75)

    # connect to camera
    print('connecting to camera...')
    cap = cv.VideoCapture(camera)  # video capture source camera (1 is external webcam)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, pixel_w)  # set desired pixel width (default 1944)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, pixel_h)  # set desired pixel height (default 2592)
    print('camera connected')

    # check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Check if webcam connected properly")  # print an error message
    else:
        start_time = time.time()  # used for waiting for autofocus
        autofocus_duration = 4  # seconds
        print('wait ' + str(autofocus_duration) + ' secs for autofocus...')  # user message

        # begin loop whilst autofocussing, only exits after autofocus time complete
        while int(time.time() - start_time) < autofocus_duration:
            ret, frame = cap.read()  # ret is bool if frame captured
            if ret == 1:
                pass
            else:
                print('error with frame capture')

        # autofocus complete and loop exits
        print('autofocus complete')

        # take image (already has autofocussed)
        ret, frame = cap.read()

        pixel_size = frame.shape[:2]  # return pixel resolution x,y
        no_of_pixels = (frame.shape[0] * frame.shape[1])/1000000  # calc resolution
        print('pixel size:', pixel_size)
        print('no. of pixels:', no_of_pixels, 'MP')

        # save image as filename.jpeg
        cv.imwrite('latest_image_from_camera' + '.jpg', frame)

    # release the cap object and close any opened windows
    cap.release()
    cv.destroyAllWindows()

    # end of function
    return 0


if __name__ == "__main__":
    camera_no = 1  # external webcam is 1 not 0
    take_picture(camera_no)
