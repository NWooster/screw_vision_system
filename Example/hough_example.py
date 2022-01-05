'''

Nathan Wooster
23/12/21git
Hough circle example algorithm

'''

import sys
import cv2 as cv
import numpy as np

#loads image, pre-process it, apply hough circle detection
def main(argv):

    '''main function called to run the vision algorithm'''

    #labels where image is
    default_file = 'smarties.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image and calls it 'src'
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    #convert image to grayscale from BGR
    #new image called 'gray'
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    #adds medium blur (5) to image to reduce noise (avoids false circle detection)
    gray = cv.medianBlur(gray, 5)

    #numpy array .shape[0] outputs the number of elements in dimension 1 of the array (number of pixel rows)
    rows = gray.shape[0]
    ##print(rows)
    ##print(gray)

    '''
    Hough circle algorithm arguments:
    
    gray: Input image (grayscale).
    circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
    HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
    dp = 1: The inverse ratio of resolution.
    min_dist = gray.rows/8: Minimum distance between detected centers.
    param_1 = 100: Upper threshold for the internal Canny edge detector.
    param_2 = 30*: Threshold for center detection.
    min_radius = 1: Minimum radius to be detected. If unknown, put zero as default.
    max_radius = 30: Maximum radius to be detected. If unknown, put zero as default.
    '''
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=1, maxRadius=30)
    ##circles_og = circles
    ##print('circles_og:', circles_og)

    #draw the detected circles
    if circles is not None:
        #removes decimals
        circles = np.uint16(np.around(circles))
        ##print('circles:', circles)
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            #draws circles (r,b,g) colour
            cv.circle(src, center, radius, (255, 0, 255), 3)

    cv.imshow("detected circles", src)

    #wait for user to press exit
    cv.waitKey(0)

    return 0

#sys.argv[1:] is the user input (empty array -  good practise)

if __name__ == "__main__":
    ##print(sys.argv[1:])
    main(sys.argv[1:])