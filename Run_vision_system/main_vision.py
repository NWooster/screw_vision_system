#!/usr/bin/env python3

'''

Nathan Wooster
17/02/22
The main python script to run screw image detection system

'''

import sys
import numpy as np

# custom imports
from take_picture import take_picture
from calibrate_camera import calibrate_camera
from screw_location import mm_screw_location


def main_vision(argv):

    """
    Main function called to run the vision algorithm.
    """

    # define cameras to connect to
    webcam = 1
    laptop_cam = 0

    #take_picture(laptop_cam, 5)  # take image from webcam (camera 1) with specified autofocus time

    # find pixel/mm ratio
    pix_to_mm, ratio_error = calibrate_camera(image_location='images_taken/with_25_square.jpg')

    # output screw locations in mm
    screw_locations, max_mm_error = mm_screw_location(pix_to_mm, ratio_error, image_location='images_taken'
                                                                                             '/with_25_square.jpg')

    # print outputs
    print('screw locations:', screw_locations)
    print()

    screw_radii = screw_locations[:, 2]
    print('screw radii:', screw_radii)
    print()

    screw_centres = screw_locations[:, :2]
    print('screw centres:', screw_centres)
    print()

    print('number of screws found: ', np.shape(screw_locations)[0])
    print()

    print('max error is: ' + str(max_mm_error) + 'mm')
    print()

    return screw_centres


if __name__ == "__main__":
    main_vision(sys.argv[1:])
