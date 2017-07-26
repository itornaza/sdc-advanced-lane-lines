
from classes.plotting import Plotting

import numpy as np
import cv2

class Convert():

    #----------
    # Methods
    #----------

    def toGray(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def toHLS(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def toUnit8(value):
        return np.uint8( (255 * value) / np.max(value) )

    def perspectiveTransform(img):
        '''Get the perspective transform of the given image'''

        # Get the image size
        img_size = (img.shape[1], img.shape[0])

        # Source vertices for perspective transform. The vertices are selected to map
        # the trapezoid that is formed from the left and right lane lines.

        # TODO: Adjust
        src_bottom_left = [220,720]
        src_bottom_right = [1110, 720]
        src_top_left = [570, 470]
        src_top_right = [722, 470]
        
        # Destination vertices for perspective transform. The vertices are mannualy
        # selected to display the lane lines nicely on a birds eye view.

        # TODO: Adjust
        dst_bottom_left = [320,720]
        dst_bottom_right = [920, 720]
        dst_top_left = [320, 1]
        dst_top_right = [920, 1]

        # Assign the source and destination vertices for the transform
        src = np.float32([src_bottom_left, src_bottom_right, src_top_right, src_top_left])
        dst = np.float32([dst_bottom_left, dst_bottom_right, dst_top_right, dst_top_left])

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        # Warp the image
        warped = cv2.warpPerspective(img, M, img_size)
        
        # Return the warped image and the perspective transform matrix
        return warped, M

    def histogramPeaks(img):
        '''
        Given an image get a histogram and the positions of the left and right lanes 
        on the x axis
        '''
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        
        # Get the middle point of the histogram
        midpoint = np.int(histogram.shape[0]/2)
        
        # Get the left peak of the histogram i.e. the left lane
        leftx_base = np.argmax(histogram[:midpoint])

        # Get the right peak of the histogram i.e. the right lane
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Plot the historgram and return the left and right lanes positions
        Plotting.plotHistogram(histogram)
        return leftx_base, rightx_base

    def slidingWindow(img):

        # Get the lane positions from the histogram
        leftx_base, rightx_base = Convert.histogramPeaks(img)
        
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img)) * 255
    
        # Choose the number of sliding windows
        nwindows = 9
        
        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set the width of the windows +/- margin
        margin = 100
        
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
        
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Plot the resulting sliding windows
        Plotting.plotSlidingWindow(out_img, left_fitx, right_fitx, ploty)
