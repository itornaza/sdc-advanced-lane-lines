
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#--------------
# Constants
#--------------

KERNEL = 7

#--------------
# Conversions
#--------------

def convertGrayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def convertHLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def convertUnit8(value):
    return np.uint8( (255 * value) / np.max(value) )

#--------------
# Sobel math
#--------------

def get_sobelx(img, kernel):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)

def get_sobely(img, kernel):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

def get_abs_sobelx(img, kernel):
    sobelx = get_sobelx(img, kernel)
    return np.absolute(sobelx)

def get_abs_sobely(img, kernel):
    sobely = get_sobely(img, kernel)
    return np.absolute(sobely)

def get_grad_mag(img, kernel):
    sobelx = get_sobelx(img, kernel)
    sobely = get_sobely(img, kernel)
    return np.sqrt(sobelx**2 + sobely**2)

def get_grad_dir(img, kernel):
    abs_sobelx = get_abs_sobelx(img, kernel)
    abs_sobely = get_abs_sobely(img, kernel)
    return np.arctan2(abs_sobely, abs_sobelx)

#--------------
# Thresholding
#--------------

def abs_sobel_thresh(gray, orient='x', kernel=3, thresh=(0, 255)):
    '''Calculate gradient'''
    
    # Take the absolute value of the derivative in x or y
    if orient == 'x':
        abs_sobel = get_abs_sobelx(gray, kernel)
    elif orient == 'y':
        abs_sobel = get_abs_sobely(gray, kernel)
    else:
        exit("Invalid input, expecting orient = x or y")
    
    # Scale to 8-bit (0 - 255) then convert to uint8
    scaled_sobel = convertUnit8(abs_sobel)

    # Create a mask of 1's where the scaled gradient magnitude is between the thresholds
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the binary mask
    return grad_binary

def mag_thresh(gray, kernel=3, mag_thresh=(0, 255)):
    '''Calculate gradient magnitude'''
    
    # Calculate the magnitude
    grad_mag = get_grad_mag(gray, kernel)
    
    # Rescale to 8 bit
    scale_factor = np.max(grad_mag) / 255
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)
    
    # Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(grad_mag)
    mag_binary[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
    
    # Return the binary mask
    return mag_binary

def dir_threshold(gray, kernel=3, thresh=(0, np.pi/2)):
    '''Calculate gradient direction'''
    
    # Calculate the direction of the gradient
    grad_dir = get_grad_dir(gray, kernel)
    
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    grad_dir = convertUnit8(grad_dir)
    
    # Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    
    # Return the binary mask
    return dir_binary

def hls_select(image, thresh=(0, 255)):
    '''Calculate the saturation channel'''
    
    # Get the HLS colorspace of the image
    hls = convertHLS(image)
    
    # Get the S-channel
    S = hls[:,:,2]
    
    # Apply thresholding
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    # Return the binary mask
    return binary_output

def hlsPlusGrad(image, kernel=3, channel_sel='s', s_thresh=(170, 255), sx_thresh=(20, 100)):

    # Convert to HLS color space and separate the S channel
    hls = convertHLS(image).astype(np.float)
    
    # Assign the channel
    channel = getHLSChannel(hls, channel_sel)
    
    # Sobel x
    abs_sobelx = get_abs_sobelx(channel, kernel)
    scaled_sobel = convertUnit8(abs_sobelx)
    
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

#--------------
# Helpers
#--------------

def getHLSChannel(hls, channel_selector):
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

def plotResult(image, combined):
    '''Plot the original and combined image side by side'''

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original', fontsize=24)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Combined', fontsize=24)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

#--------------
# Main
#--------------

if __name__ == '__main__':
    
    # Read in an image
    image = mpimg.imread('signs_vehicles_xygrad.png')

    # Convert to grayscale and run the thresholding functions on the grayscale image
    gray = convertGrayscale(image)

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', kernel=KERNEL, thresh=(30, 100))
    grady = abs_sobel_thresh(gray, orient='y', kernel=KERNEL, thresh=(30, 100))
    mag_binary = mag_thresh(gray, kernel=KERNEL, mag_thresh=(90, 110))
    dir_binary = dir_threshold(gray, kernel=KERNEL, thresh=(0.7, np.pi/2))

    # Combine the thresholding functions into one
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Plot the original and combined images
    plotResult(image, combined)

    # Plot stacked images from pipeline
    plotResult(image, hlsPlusGrad(image, KERNEL))
