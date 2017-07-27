
from classes.image_processing import Image_processing
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
test_image = 'test_images/test2.jpg' # straight_lines1

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
    undist = Camera.undistort(image, mtx, dist)
    
    # Process the undistorted image with thresholding to get a binary mask
    binary_mask = Thresholding.lane_detection_mask(undist, KERNEL)
    
    # Plot the undistorted and binary mask images
    Plotting.plotResult(undist, binary_mask)

    # Perspective transform
    warped, M, M_inv = Image_processing.perspectiveTransform(binary_mask)
    Plotting.plotResult(binary_mask, warped)

    # Get the position of the left and right lanes
    leftx_base, rightx_base = Image_processing.histogramPeaks(warped)
    left_fit, right_fit = Image_processing.slidingWindowInit(warped)

    # Calculate the sliding window of a successive image, using the lane line
    # equations that are already known from the previous image analysis
    left_fit, right_fit = Image_processing.slidingWindowFollowing(warped, left_fit, right_fit)

    # Calculate the curvature of the left and right lanes
    left_curv, right_curv, curv_string= Image_processing.curvature(warped, left_fit, right_fit)

    # Calculate the offset of the car from the center of the lane line
    offset_m, offset_string = Image_processing.offsetFromCenter(undist, leftx_base, rightx_base)

    # Overlay the lane area to the undistorted image
    Image_processing.laneArea(warped, undist, M_inv, left_fit, right_fit, offset_string)

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
