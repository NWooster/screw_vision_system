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


def main(argv):
    """
    Main function called to run the vision algorithm.
    """

    # take_picture(0)  # take image from webcam (camera 1)
    calibrate_camera(7, 7, 25, 25)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
