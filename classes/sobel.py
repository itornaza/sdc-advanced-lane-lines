
import numpy as np
import cv2

class Sobel():

    def get_x(img, kernel):
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)

    def get_y(img, kernel):
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

    def get_abs_x(img, kernel):
        sobelx = Sobel.get_x(img, kernel)
        return np.absolute(sobelx)

    def get_abs_y(img, kernel):
        sobely = Sobel.get_y(img, kernel)
        return np.absolute(sobely)

    def get_grad_mag(img, kernel):
        sobelx = Sobel.get_x(img, kernel)
        sobely = Sobel.get_y(img, kernel)
        return np.sqrt(sobelx**2 + sobely**2)

    def get_grad_dir(img, kernel):
        abs_sobelx = Sobel.get_abs_x(img, kernel)
        abs_sobely = Sobel.get_abs_y(img, kernel)
        return np.arctan2(abs_sobely, abs_sobelx)
