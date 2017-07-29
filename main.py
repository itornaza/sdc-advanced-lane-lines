
from classes.camera import Camera
from classes.image_processing import Image_processing
from classes.line import Line
from classes.plotting import Plotting
from classes.thresholding import Thresholding

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from enum import Enum
from moviepy.editor import VideoFileClip

#--------------
# Globals
#--------------

left_lane = Line()
right_lane = Line()
initiate_sw = True  # Control variable for the sliding window technique to use

# Constants

KERNEL = 7 # Increase for smoother result (odd numbers only)
test_image = 'test_images/test2.jpg' # straight_lines1
video_in = 'project_video.mp4'
video_out = 'project_video_output.mp4'

class Commands(Enum):
    NONE = 0
    CALIBRATE = 1
    EXPLORE = 2

#--------------
# Pipeline
#--------------

def exploratory_pipeline():
    '''
    The pipeline to calculate the lane lines in an exploratory mode that produces
    plots of the various stages along the way.
    
    Notes:
    - Calibration shall be performed at least once before using the pipeline
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
    leftx_base, rightx_base = Image_processing.histogramPeaks(warped, plot=True)

    # Get the lane line equations using the sliding window
    left_fit, right_fit = Image_processing.slidingWindowInit(warped, plot=True)

    # Calculate the sliding window of a successive image, using the lane line
    # equations that are already known from the previous image analysis
    left_fit, right_fit = Image_processing.slidingWindowFollowing(warped, left_fit,
                                                                  right_fit, plot=True)

    # Calculate the curvature of the left and right lanes
    left_curv, right_curv, curv_string= Image_processing.curvature(warped, left_fit, right_fit)

    # Calculate the offset of the car from the center of the lane line
    offset_m, offset_string = Image_processing.offsetFromCenter(undist, leftx_base, rightx_base)

    # Overlay the lane area to the undistorted image
    Image_processing.laneArea(warped, undist, M_inv, left_fit, right_fit,
                              offset_string, plot=True)

def pipeline(image):
    '''
    The lane detection pipeline
    
    Notes:
    - Calibration shall be performed at least once before using the pipeline
    '''
    global left_lane
    global right_lane
    global initiate_sw
    
    # Undistort the image before processing and plot an example
    mtx, dist = Camera.getCalibrationData()
    undist = Camera.undistort(image, mtx, dist)
    
    # Process the undistorted image with thresholding to get a binary mask
    binary_mask = Thresholding.lane_detection_mask(undist, KERNEL)
    
    # Perspective transform
    warped, M, M_inv = Image_processing.perspectiveTransform(binary_mask)
    
    # Get the position of the left and right lanes
    leftx_base, rightx_base = Image_processing.histogramPeaks(warped)




    # Apply the appropriate slidinig window technique
    if initiate_sw:
        # Get the lane line equations using the sliding window
        left_fit, right_fit = Image_processing.slidingWindowInit(warped)
        initiate_sw = False
    else:
        # Calculate the sliding window of a successive image, using the lane line
        # equations that are already known from the previous image analysis
        left_fit, right_fit = Image_processing.slidingWindowFollowing(warped, \
                                    left_lane.current_fit, \
                                    right_lane.current_fit)

        # Recover from loosing lines with initializing a new window search
        # if the line second order coefficient is smaller than a margin
        if left_fit[0] < 1.0e-04 or right_fit[0] < 1.0e-04:
            initiate_sw = True

    # Calculate the curvature of the left and right lanes
    left_curv, right_curv, curv_string = Image_processing.curvature(warped, left_fit, right_fit)
    
    # Calculate the offset of the car from the center of the lane line
    offset_m, offset_string = Image_processing.offsetFromCenter(undist, leftx_base, rightx_base)
    
    # Overlay the lane area to the undistorted image
    result = Image_processing.laneArea(warped, undist, M_inv, left_fit, right_fit, offset_string)
    
    # Update the globals
    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit

    # Return processed image
    return result

def createVideo(video_in, video_out):
    '''Take a video as an imput and run the lane detection pipeline on it'''
    
    clip = VideoFileClip(video_in)
    white_clip = clip.fl_image(pipeline).subclip(20, 26)
    white_clip.write_videofile(video_out, audio=False)

#--------------
# Main
#--------------

def help():
    print()
    print("> -c: Calibrate the camera")
    print("> -e: Explore pipeline and plot inner stages")
    print("> Do not enter flag to run the pipeline")
    print()

def parseCommands():
    '''Parse the command line arguments'''

    command = Commands.NONE
    if len(sys.argv) > 1:
        if sys.argv[1] == '-c':
            command = Commands.CALIBRATE
        elif sys.argv[1] == '-e':
            command = Commands.EXPLORE
        else:
            help()
            exit(0)

    return command

if __name__ == '__main__':
    command = parseCommands()
    if command == Commands.CALIBRATE:
        Camera.calibrate()
    elif command == Commands.EXPLORE:
        exploratory_pipeline()
    else:
        createVideo(video_in, video_out)
