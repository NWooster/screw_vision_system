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

    take_picture.take_picture(webcam, 10)  # take image from webcam (camera 1) with specified autofocus time

    # find pixel/mm ratio
    pix_to_mm, tl_corner_pix, ratio_error = calibrate_camera.calibrate_camera(image_location='images_taken/'
                                                                                             '1latest_image_from_'
                                                                                             'camera.jpg')

    # output screw locations in mm
    screw_locations, max_mm_error = screw_location.mm_screw_location(pix_to_mm, tl_corner_pix, ratio_error,
                                                                     image_location='images_taken/'
                                                                                    '1latest_image_from_camera.jpg')
    # select centres only
    screw_centres = screw_locations[:, :2]

    # Print outputs
    # print('screw locations:', screw_locations)
    # print()
    # screw_radii = screw_locations[:, 2]
    # print('screw radii:', screw_radii)
    # print()
    # print('screw centres:', screw_centres)
    # print()
    # print('number of screws found: ', np.shape(screw_locations)[0])
    # print()

    # Results
    print('Resolution is ' + str(pix_to_mm) + 'mm (1 pixel is ' + str(pix_to_mm) + 'mm)')  # resolution

    print('max error due to mm to pixel conversion error is: ' + str(max_mm_error) + 'mm')
    print()
    print('------------------------------')

    return screw_centres


if __name__ == "__main__":

    # what next code will do
    all_screw_coords = main_vision(sys.argv[0])  # runs vision algorithm
    all_screw_coords = all_screw_coords[all_screw_coords[:, 1].argsort()]  # re-order based on y coordinate
    print(all_screw_coords)
    print()

    screw1_coord = all_screw_coords[0:]  # selects first screw coordinate
    screw1_coord = np.append(screw1_coord[0, :], [0])  # appends a 0 for the Z axis

    print('screw_coord', screw1_coord)
    print()
    print('Add this coordinate to the constant (x,y) distance of gantry (0,0) to the INNER top left corner to get '
          'where to move gantry to in mm.')
