
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

#--------------
# Pipeline
#--------------

def pipeline(calibrate):
    '''The pipeline to calculate the lane lines'''
    
    # Calibrate the camera if the "-cal" option is provided
    if calibrate: Camera.calibrate()
    
    # Read in an image
    image = mpimg.imread('test_images/straight_lines1.jpg')
    
    # Undistort the image before processing and plot an example
    mtx, dist = Camera.getCalibrationData()
    undistorted_image = Camera.undistort(image, mtx, dist)
    Plotting.plotUndistortedImage(image, undistorted_image)
    
    # Process the undistorted image with thresholding to get a binary mask
    binary_mask = Thresholding.hlsPlusGrad(undistorted_image, KERNEL)
    #binary_mask = Thresholding.sobelGrayscaleCombo(image, KERNEL)
    
    # Plot the undistorted and binary mask images
    Plotting.plotResult(undistorted_image, binary_mask)

#--------------
# Main
#--------------

def parseCommands():
    '''Parse the command line arguments'''
    if len(sys.argv) > 1:
        if sys.argv[1] == '-cal':
            return True
        else:
            print("Enter -cal to calibrate the camera")
            exit(0)
    else:
        return False

if __name__ == '__main__':
    calibrate = parseCommands()
    pipeline(calibrate)
