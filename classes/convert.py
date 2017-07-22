

import numpy as np
import cv2

#--------------
# Conversions
#--------------

class Convert():

    def toGray(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def toHLS(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def toUnit8(value):
        return np.uint8( (255 * value) / np.max(value) )
