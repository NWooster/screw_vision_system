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
    # define camera
    cap = cv.VideoCapture(camera)  # video capture source camera (Here external webcam)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 2448)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 3264)
    print('camera connected')

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Check if webcam connected properly")  # print and error message
    else:
        start_time = time.time()
        print(start_time)
        capture_duration = 10

        while (int(time.time() - start_time) < capture_duration):
            ret, frame = cap.read()
            if ret == True:
                print('autofocus in progress')
            else:
                print('error with frame')

        print('final time:',time.time()-start_time)

        ret, frame = cap.read()

        pixel_size = frame.shape[:2]
        print('pixel size:',pixel_size)

        cv.imwrite('image_from_new_cam' + '.jpg', frame)


    # After the loop release the cap object
    cap.release()
    #out.release()
    # Destroy all the windows
    cv.destroyAllWindows()

#    if cv.waitKey(1) & 0xFF == ord('q'):
 #       cap.release()  # release the camera object
  #      cv.destroyAllWindows()  # close all other windows

    return 0


if __name__ == "__main__":
    camera_no = 1
    take_picture(camera_no)
