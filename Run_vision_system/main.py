#!/usr/bin/env python3

'''

Nathan Wooster
17/02/22
The main python script to run screw image detection system

'''

import sys

# custom imports
from take_picture import take_picture
from calibrate_camera import calibrate_camera
from screw_location import screw_location


def main(argv):
    """
    Main function called to run the vision algorithm.
    """

    # take_picture(0)  # take image from webcam (camera 1)
    pix_to_mm, ratio_error = calibrate_camera(image_location='images_taken/with_25_square.jpg')
    screw_location(image_location='images_taken/with_25_square.jpg')

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
