
from classes.convert import Convert
from classes.thresholding import Thresholding
from classes.plotting import Plotting
from classes.camera import Camera

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#--------------
# Constants
#--------------

KERNEL = 7 # Increase for smoother result (odd numbers only)
test_image = 'test_images/straight_lines1.jpg'

#--------------
# Pipeline
#--------------

def pipeline():
    '''
    The pipeline to calculate the lane lines.
    Calibration shall be performed at least once before using the pipeline
    '''
    # Read in an image
    image = mpimg.imread(test_image)
    
    # Undistort the image before processing and plot an example
    mtx, dist = Camera.getCalibrationData()
    undistorted_image = Camera.undistort(image, mtx, dist)
    
    # Process the undistorted image with thresholding to get a binary mask
    binary_mask = Thresholding.lane_detection_mask(undistorted_image, KERNEL)
    
    # Plot the undistorted and binary mask images
    Plotting.plotResult(undistorted_image, binary_mask)

    # Perspective transform
    warped, M_perspective = Convert.perspectiveTransform(binary_mask)
    Plotting.plotResult(binary_mask, warped)

    # Get the position of the left and right lanes
    leftx_base, rightx_base = Convert.histogramPeaks(warped)
    Convert.slidingWindow(warped)

    # Fit

#--------------
# Main
#--------------

def parseCommands():
    '''Parse the command line arguments'''
    if len(sys.argv) > 1:
        if sys.argv[1] == '-c':
            return True
        else:
            print("Enter -c to calibrate the camera")
            exit(0)
    else:
        return False

if __name__ == '__main__':
    calibrate = parseCommands()
    if calibrate:
        Camera.calibrate()
    else:
        pipeline()
