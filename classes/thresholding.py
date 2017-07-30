
from classes.sobel import Sobel
from classes.image_processing import Image_processing

import numpy as np
import cv2

class Thresholding():

    #------------
    # API
    #------------

    def abs_sobel(gray, orient='x', kernel=3, thresh=(0, 255)):
        '''Calculate gradient'''
        
        # Take the absolute value of the derivative in x or y
        if orient == 'x':
            abs_sobel = Sobel.get_abs_x(gray, kernel)
        elif orient == 'y':
            abs_sobel = Sobel.get_abs_y(gray, kernel)
        else:
            exit("Invalid input, expecting orient = x or y")
        
        # Scale to 8-bit (0 - 255) then convert to uint8
        scaled_sobel = Image_processing.toUnit8(abs_sobel)

        # Create a mask of 1's where the scaled gradient magnitude is between the thresholds
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the binary mask
        return grad_binary

    def mag(gray, kernel=3, mag_thresh=(0, 255)):
        '''Calculate gradient magnitude'''
        
        # Calculate the magnitude
        grad_mag = Sobel.get_grad_mag(gray, kernel)
        
        # Rescale to 8 bit
        scale_factor = np.max(grad_mag) / 255
        grad_mag = (grad_mag/scale_factor).astype(np.uint8)
        
        # Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(grad_mag)
        mag_binary[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
        
        # Return the binary mask
        return mag_binary

    def dir(gray, kernel=3, thresh=(0, np.pi/2)):
        '''Calculate gradient direction'''
        
        # Calculate the direction of the gradient
        grad_dir = Sobel.get_grad_dir(gray, kernel)
        
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        grad_dir = Image_processing.toUnit8(grad_dir)
        
        # Create a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(grad_dir)
        dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
        
        # Return the binary mask
        return dir_binary

    def gray_masks(image, kernel):
        '''Combines sobel, magnitude and direction methods to create a binary mask'''
        
        # Convert to grayscale and run the thresholding functions on the grayscale image
        gray = Image_processing.toGray(image)
        
        # Apply each of the thresholding functions
        gradx = Thresholding.abs_sobel(gray, orient='x', kernel=kernel, thresh=(30, 100))
        grady = Thresholding.abs_sobel(gray, orient='y', kernel=kernel, thresh=(30, 100))
        mag_binary = Thresholding.mag(gray, kernel=kernel, mag_thresh=(90, 110))
        dir_binary = Thresholding.dir(gray, kernel=kernel, thresh=(0.7, np.pi/2))
        
        # Combine the thresholding functions into one
        gray_mask = np.zeros_like(dir_binary)
        gray_mask[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
        # Return binary mask
        return gray_mask

    def hls_masks(image, kernel):
        '''Get binary masks from processing the HLS colorspace'''
        
        # Set up threshold values for the channels
        s_thresh = (170, 255)
        sx_thresh = (20, 100)
        
        # Convert to HLS color space and separate the S channel
        hls = Image_processing.toHLS(image)
        
        # Get the S channel
        S = Thresholding._getHLSChannel(hls, 's')
        
        # Sobel x
        abs_sobelx = Sobel.get_abs_x(S, kernel)
        scaled_sobel = Image_processing.toUnit8(abs_sobelx)
        
        # Threshold S x gradient
        sx_mask = np.zeros_like(scaled_sobel)
        sx_mask[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold S
        s_mask = np.zeros_like(S)
        s_mask[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1

        # Return binary mask
        return s_mask, sx_mask

    def rgb_masks(image, kernel):
        '''Get binary masks from processing the RGB colorspace'''

        # Set up threshold values for the channels
        r_thresh  = (150, 255)
        
        # Get the R channel
        R = Thresholding._getRGBChannel(image, 'r')
        
        # Threshold R
        r_mask = np.zeros_like(R)
        r_mask[(R >= r_thresh[0]) & (R <= r_thresh[1])] = 1

        # Return binary mask
        return r_mask
    
    def l_mask(image, kernel):
        '''
        Get binary masks from processing the L channel of HLS colorspace
        to take care of the shadow areas
        '''
        
        # Set up threshold values for the channels
        l_thresh  = (120, 255)
        
        # Get the L channel
        L = Thresholding._getHLSChannel(image, 'l')
        
        # Threshold L
        l_mask = np.zeros_like(L)
        l_mask[(L >= l_thresh[0]) & (L <= l_thresh[1])] = 1

        # Return binary mask
        return l_mask
    
    def region_of_interest(image):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        
        # Get the dimensions of the image
        x_size = image.shape[1]
        y_size = image.shape[0]
        
        # Assign the vertices of the region of interest
        vertices = np.array([[
                              (x_size*(1/24), y_size),
                              (x_size*(11/24), y_size*(14/24)),
                              (x_size*(13/24), y_size*(14/24)),
                              (x_size*(23/24), y_size)]],
                            dtype=np.int32)

        
        # Defining a blank mask to start with
        mask = np.zeros_like(image)
        
        # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        
        # Filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        # Returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def lane_detection_mask(image, kernel):
        '''
        Creates a robust mask to detect lane lines of white or yellow color in shades
        under daylight conditions
        '''
        
        # Read in the image
        image = Thresholding.region_of_interest(image)
            
        # HLS processing
        s_mask, sx_mask = Thresholding.hls_masks(image, kernel)
        
        # Avoid pixels which have shadows and as a result darker.
        l_mask = Thresholding.l_mask(image, kernel)

        # RGB processing
        r_mask = Thresholding.rgb_masks(image, kernel)
            
        # Grayscale processing
        gray_mask = Thresholding.gray_masks(image, kernel)
                
        # Return binary mask
        return ((s_mask & sx_mask) | gray_mask) & (r_mask & l_mask)
    
    #------------
    # Methods
    #------------

    def _getHLSChannel(hls, channel_selector):
        '''Given an hls colorspace and a channel, returns the appropriate channel'''
        
        # Channels matrix indices
        H = 0
        L = 1
        S = 2
        
        # Get the channel from the selector
        if channel_selector == 'h':
            channel = hls[:,:,H]
        elif channel_selector == 'l':
            channel = hls[:,:,L]
        else:
            channel = hls[:,:,S]
    
        # Return the channel
        return channel

    def _getRGBChannel(rgb, channel_selector):
        '''Given an rgb colorspace and a channel, returns the appropriate channel'''
        
        # Channels matrix indices
        R = 0
        G = 1
        B = 2
        
        # Get the channel from the selector
        if channel_selector == 'r':
            channel = rgb[:,:,R]
        elif channel_selector == 'g':
            channel = rgb[:,:,G]
        else:
            channel = rgb[:,:,B]
        
        # Return the channel
        return channel

