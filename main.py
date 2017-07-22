
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

KERNEL = 7

#--------------
# Main
#--------------

if __name__ == '__main__':
    
    # Calibrate the camera and create a calibration file
    Camera.pipeline()
    
    # Read in an image
    image = mpimg.imread('signs_vehicles_xygrad.png')

    # Convert to grayscale and run the thresholding functions on the grayscale image
    gray = Convert.toGray(image)

    # Apply each of the thresholding functions
    gradx = Thresholding.abs_sobel(gray, orient='x', kernel=KERNEL, thresh=(30, 100))
    grady = Thresholding.abs_sobel(gray, orient='y', kernel=KERNEL, thresh=(30, 100))
    mag_binary = Thresholding.mag(gray, kernel=KERNEL, mag_thresh=(90, 110))
    dir_binary = Thresholding.dir(gray, kernel=KERNEL, thresh=(0.7, np.pi/2))

    # Combine the thresholding functions into one
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Plot the original and combined images
    Plotting.plotResult(image, combined)

    # Plot stacked images from pipeline
    Plotting.plotResult(image, Thresholding.hlsPlusGrad(image, KERNEL))
