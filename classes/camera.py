
from classes.plotting import Plotting

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

class Camera():

    rows = 6
    cols = 9
    images_path = 'camera_cal/calibration*.jpg'
    test_image_path = 'camera_cal/calibration13.jpg'
    delay = 500
    image_string = 'img'

    def prepareObjects():
        '''Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)'''
        
        objp = np.zeros((Camera.rows * Camera.cols, 3), np.float32)
        objp[:,:2] = np.mgrid[0:Camera.cols, 0:Camera.rows].T.reshape(-1,2)
        
        return objp

    def collectObjectPoints(objp):
    
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
    
        # Make a list of calibration images
        images = glob.glob(Camera.images_path)
            
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (Camera.cols, Camera.rows), None)
            
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (Camera.cols, Camera.rows), corners, ret)
                cv2.imshow(Camera.image_string, img)
                cv2.waitKey(Camera.delay)

        return objpoints, imgpoints

    def saveUndistortedImage(img, mtx, dist):
        '''# Save the undistorted test image'''
        
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite('camera_cal/test_undist.jpg', dst)
    
        return dst

    def saveCalibration(mtx, dist):
        '''Save the camera calibration result for later use'''

        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( "camera_cal/dist_pickle.p", "wb" ) )

    def calibrate(objpoints, imgpoints):
        '''Do camera calibration given object points and image points'''
        
        img = cv2.imread(Camera.test_image_path)
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
        return img, mtx, dist

    def pipeline():
        objp = Camera.prepareObjects()
        objpoints, imgpoints = Camera.collectObjectPoints(objp)
        cv2.destroyAllWindows()
        img, mtx, dist = Camera.calibrate(objpoints, imgpoints)
        dst = Camera.saveUndistortedImage(img, mtx, dist)
        Camera.saveCalibration(mtx, dist)
        Plotting.plotUndistortedImage(img, dst)
