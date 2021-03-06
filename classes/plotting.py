
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Plotting():

    def plotResult(image, combined):
        '''Plot the original and combined image side by side'''
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original', fontsize=24)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Binary mask', fontsize=24)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    def plotUndistortedImage(image, dst):
        '''Visualize undistortion'''
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.show()

    def plotHistogram(histogram):
        plt.plot(histogram)
        plt.show()

    def plotSlidingWindow(out_img, left_fitx, right_fitx, ploty):
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    def simplePlot(image):
        plt.imshow(image)
        plt.show()
