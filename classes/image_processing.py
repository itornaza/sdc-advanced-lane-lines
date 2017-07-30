
from classes.plotting import Plotting

import numpy as np
import cv2

class Image_processing():

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
        src_bottom_left = [220,720]
        src_bottom_right = [1110, 720]
        src_top_left = [570, 470]
        src_top_right = [720, 470]
        
        # Destination vertices for perspective transform. The vertices are mannualy
        # selected to display the lane lines nicely on a birds eye view.
        dst_bottom_left = [320,720]
        dst_bottom_right = [920, 720]
        dst_top_left = [320, 1]
        dst_top_right = [920, 1]

        # Assign the source and destination vertices for the transform
        src = np.float32([src_bottom_left, src_bottom_right, src_top_right, src_top_left])
        dst = np.float32([dst_bottom_left, dst_bottom_right, dst_top_right, dst_top_left])

        # Given src and dst points, calculate the perspective transform and the
        # inverse perspective transform matrices
        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        
        # Warp the image
        warped = cv2.warpPerspective(img, M, img_size)
        
        # Return the warped image and the perspective transform matrix
        return warped, M, M_inv

    def histogramPeaks(img, plot=False):
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
        if plot: Plotting.plotHistogram(histogram)
        return leftx_base, rightx_base

    def slidingWindowInit(img, plot=False):
        '''
        Does the sliding window processing on an image and returns the second order
        polynomials for the left and right lane
        '''

        # Get the lane positions from the histogram
        leftx_base, rightx_base = Image_processing.histogramPeaks(img)
        
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
        
        # Visualize the result
        if plot:
            # Generate x and y values for plotting
            left_fitx, right_fitx, ploty = Image_processing.equations2components(img, left_fit, right_fit)
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Plot the resulting sliding windows
            Plotting.plotSlidingWindow(out_img, left_fitx, right_fitx, ploty)

        # Return the lane lines equations
        return left_fit, right_fit

    def slidingWindowFollowing(img, left_fit, right_fit, plot=False):
        '''
        Does the sliding window processing on an image and returns the second order
        polynomials for the left and right lane. It assumes that the sliding window of
        a previous image is already calculated and the left and right lane lines 
        second order polynomials are provided.
        '''

        # Assume you now have a new warped binary image from the next frame of video
        # (also called "img"). It's now much easier to find line pixels!
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) +
                                       left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0] * (nonzeroy**2) +
                                       left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) +
                                        right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0] * (nonzeroy**2) +
                                        right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        left_fitx, right_fitx, ploty = Image_processing.equations2components(img, left_fit, right_fit)

        # Visualize the results
        if plot:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((img, img, img)) * 255
            window_img = np.zeros_like(out_img)
            
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the resulting image of the lane lines
            Plotting.plotSlidingWindow(result, left_fitx, right_fitx, ploty)

        # Return the lane lines equations
        return left_fit, right_fit

    def equations2components(img, left_fit, right_fit):
        '''Takes the lane line equations and returns the x and y components'''
        
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        # Return the x and y components
        return left_fitx, right_fitx, ploty

    def curvature(img, left_fit, right_fit):
        '''
        Calculate the curvature of the left and right lane lines given their line equations
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30.0 / 720.0 # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700.0 # meters per pixel in x dimension

        # Get the x and y components
        left_fitx, right_fitx, ploty = Image_processing.equations2components(img, left_fit, right_fit)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        ploty_cr = ploty * ym_per_pix
        y_eval = np.max(ploty_cr)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        
        # Build a reporting string for curvature displaying the minimum of the two
        curvature = min(left_curverad, right_curverad)
        curvature_string = "Radius of curvature: %.2f m" % curvature
        
        # Return the radius of curvature in meters
        return left_curverad, right_curverad, curvature_string

    def offsetFromCenter(img, leftx_base, rightx_base):
        '''
        Calculate the offset of the car from the center of the lane line in meters
        '''
        xm_per_pix = 3.7 / 700.0
        lane_center = (leftx_base + rightx_base) / 2.0
        offset_p = abs(img.shape[1]/2 - lane_center)
        offset_m = offset_p * xm_per_pix
        offset_string = "Offset: %.2f m" % offset_m

        # Return the offset
        return offset_m, offset_string

    def laneArea(warped, undist, M_inv, left_fit, right_fit, offset_string, plot=False):
        '''
        Overlays the undistorted image with the area that is detected as lane in a single plot
        '''
        
        # Get the x and y components
        left_fitx, right_fitx, ploty = Image_processing.equations2components(warped, left_fit, right_fit)
        
        # Calculate curvature
        left_curv, right_curv, curv_string = Image_processing.curvature(warped, left_fit, right_fit)
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, M_inv, (undist.shape[1], undist.shape[0]))
        
        # Overlay the curvature
        cv2.putText(undist, curv_string , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
        
        # Overlay the car offset from the center of the lane line
        cv2.putText(undist, offset_string, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
        
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        if plot: Plotting.simplePlot(result)

        # Return the image overlayed with the lane line detectio and curvature info
        return result

    def sanityChecks(img, left_fit, right_fit):
        '''
        If all the individual checks for similar curvature and lateral distance are passed
        then returns true
        '''
        
        curvature_margin = 1000
        distance_margin = 201
        
        # Check for have similar curvature
        check_curv = True
        left_curv, right_curv, string = Image_processing.curvature(img, left_fit, right_fit)
        if abs(left_curv - right_curv) > curvature_margin:
            check_curv = False
        
        # Check lateral separation
        check_distance = True
        if np.mean(right_fit - left_fit) > distance_margin:
            check_distance = False
        
        return check_curv and check_distance

    def averageLines(left_lane, right_lane, n):
        '''Performs an average on the last n lanes'''

        if len(left_lane.recent_fitted) < n:
            left_lane.recent_fitted.append(left_fit)
            right_lane.recent_fitted.append(right_fit)
            left_lane.best_fit = left_fit
            right_lane.best_fit = right_fit

        else:
            for ix in range(1, n):
                left_lane.recent_fitted[ix - 1]  = left_lane.recent_fitted[ix]
                right_lane.recent_fitted[ix - 1]  = right_lane.recent_fitted[ix]
        
            left_lane.recent_fitted[n - 1] = left_fit
            right_lane.recent_fitted[n - 1] = right_fit
            
            # Zeroise contents
            left_lane.best_fit[0] = 0
            left_lane.best_fit[1] = 0
            left_lane.best_fit[2] = 0
            right_lane.best_fit[0] = 0
            right_lane.best_fit[1] = 0
            right_lane.best_fit[2] = 0
            
            # Recalculate averaged coefficients
            for ix in range(0, n):
                left_lane.best_fit[0] += left_lane.recent_fitted[ix][0]
                left_lane.best_fit[1] += left_lane.recent_fitted[ix][1]
                left_lane.best_fit[2] += left_lane.recent_fitted[ix][2]
                right_lane.best_fit[0] += right_lane.recent_fitted[ix][0]
                right_lane.best_fit[1] += right_lane.recent_fitted[ix][1]
                right_lane.best_fit[2] += right_lane.recent_fitted[ix][2]
            
            left_lane.best_fit[0] = left_lane.best_fit[0] / 6
            left_lane.best_fit[1] = left_lane.best_fit[1] / 6
            left_lane.best_fit[2] = left_lane.best_fit[2] / 6
            right_lane.best_fit[0] = right_lane.best_fit[0] / 6
            right_lane.best_fit[1] = right_lane.best_fit[1] / 6
            right_lane.best_fit[2] = right_lane.best_fit[2] / 6

        return left_lane, right_lane
