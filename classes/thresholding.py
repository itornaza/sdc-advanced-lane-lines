
from classes.sobel import Sobel
from classes.convert import Convert

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
        scaled_sobel = Convert.toUnit8(abs_sobel)

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
        grad_dir = Convert.toUnit8(grad_dir)
        
        # Create a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(grad_dir)
        dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
        
        # Return the binary mask
        return dir_binary

    def hls(image, thresh=(0, 255)):
        '''Calculate the saturation channel'''
        
        # Get the HLS colorspace of the image
        hls = Convert.toHLS(image)
        
        # Get the S-channel
        S = hls[:,:,2]
        
        # Apply thresholding
        binary_output = np.zeros_like(S)
        binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
        
        # Return the binary mask
        return binary_output

    def sobelGrayscaleCombo(image, kernel):
        '''Combines sobel, magnitude and direction methods to create a binary mask'''
        
        # Convert to grayscale and run the thresholding functions on the grayscale image
        gray = Convert.toGray(image)
        
        # Apply each of the thresholding functions
        gradx = Thresholding.abs_sobel(gray, orient='x', kernel=kernel, thresh=(30, 100))
        grady = Thresholding.abs_sobel(gray, orient='y', kernel=kernel, thresh=(30, 100))
        mag_binary = Thresholding.mag(gray, kernel=kernel, mag_thresh=(90, 110))
        dir_binary = Thresholding.dir(gray, kernel=kernel, thresh=(0.7, np.pi/2))
        
        # Combine the thresholding functions into one
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
        # Return binary mask
        return combined


    def hlsPlusGrad(image, kernel=3, channel_sel='s', s_thresh=(170, 255), sx_thresh=(20, 100)):
        '''Return the mask binary from processing both the HLS color space and one of it's channels'''

        # Convert to HLS color space and separate the S channel
        hls = Convert.toHLS(image).astype(np.float)
        
        # Assign the channel
        channel = Thresholding._getHLSChannel(hls, channel_sel)
        
        # Sobel x
        abs_sobelx = Sobel.get_abs_x(channel, kernel)
        scaled_sobel = Convert.toUnit8(abs_sobelx)
        
        # Threshold channel x gradient
        sx_binary = np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold channel
        s_binary = np.zeros_like(channel)
        s_binary[(channel >= s_thresh[0]) & (channel <= s_thresh[1])] = 1

        # Stack channels
        channel_binary = np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary))

        # Return binary mask
        return channel_binary

    def optimum(image, kernel):
        '''
        Creates a robust mask to detect lane lines of white or yellow color in shades 
        under daylight conditions
        '''
        # TODO: Get the S channel from the HSL color space
        
        # TODO: Get the R channel from the RGB color space
        
        # TODO: Get the direction to focus on vertical lines
        
        # TODO: Combine the above masks into one
        
        # TODO: Return the binary mask

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

