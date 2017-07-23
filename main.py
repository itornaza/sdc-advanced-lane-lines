
from classes.convert import Convert
from classes.thresholding import Thresholding
from classes.plotting import Plotting
from classes.camera import Camera

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#--------------
# Constants
#--------------

CALIBRATE = False # Set to True the first time you run the program
KERNEL = 7 # Increase for smoother result (odd numbers only)

#--------------
# Pipeline
#--------------

def pipeline():
    # Calibrate the camera and create a calibration file
    if CALIBRATE: Camera.calibrate()
    
    # Read in an image
    image = mpimg.imread('test_images/straight_lines1.jpg')
    
    # TODO: Undistort the image before processing
    
    # Get the binary mask of the image after sobel processing
    combined = Thresholding.hlsPlusGrad(image, KERNEL)
    #combined = Thresholding.sobelGrayscaleCombo(image, KERNEL)
    
    # Plot the original and combined images
    Plotting.plotResult(image, combined)

if __name__ == '__main__':
    pipeline()
