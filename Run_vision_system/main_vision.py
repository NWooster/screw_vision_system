#!/usr/bin/env python3

'''

Nathan Wooster
17/02/22
The main python script to run screw image detection system

'''

import sys
import numpy as np

# custom imports
import take_picture
import calibrate_camera
import screw_location


def main_vision(argv):
    """
    Main function called to run the vision algorithm.
    """

    # define cameras to connect to
    webcam = 1
    laptop_cam = 0

    # take_picture.take_picture(laptop_cam, 5)  # take image from webcam (camera 1) with specified autofocus time

    # find pixel/mm ratio
    pix_to_mm, ratio_error = calibrate_camera.calibrate_camera(image_location='images_taken/'
                                                                              '1latest_image_from_camera.jpg')

    # output screw locations in mm
    screw_locations, max_mm_error = screw_location.mm_screw_location(pix_to_mm, ratio_error,
                                                                     image_location='images_taken/'
                                                                                    '1latest_image_from_camera.jpg')

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
    screw_coords = main_vision(sys.argv[1:])
    screw_coord = np.append(screw_coords[0, :], [0])
    print('screw_coord', screw_coord)
