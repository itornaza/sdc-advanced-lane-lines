
from classes.plotting import Plotting

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

class Camera():

    #-------------
    # Properties
    #-------------
    
    rows = 6
    cols = 9
    images_path = 'camera_cal/calibration*.jpg'
    test_image_path = 'camera_cal/calibration3.jpg'
    test_undistort_image_path = 'output_images/test_undist_calibration3.jpg'
    calibration_archive = "camera_cal/dist_pickle.p"
    delay = 500
    image_string = 'img'
    img = cv2.imread(test_image_path)
    mtx_string = 'mtx'
    dist_string = 'dist'
    
    #-------------
    # Methods
    #-------------

    def _prepareObjects():
        '''Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)'''
        objp = np.zeros((Camera.rows * Camera.cols, 3), np.float32)
        objp[:,:2] = np.mgrid[0:Camera.cols, 0:Camera.rows].T.reshape(-1,2)
        return objp

    def _collectObjectPoints(objp):
        '''Calculate the object points and image points from all the images'''
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
                
                # Draw and display the corners for each image to visually test precision
                cv2.drawChessboardCorners(img, (Camera.cols, Camera.rows), corners, ret)
                cv2.imshow(Camera.image_string, img)
                cv2.waitKey(Camera.delay)

        # Close all chessboard plots
        cv2.destroyAllWindows()

        # Return the calibration points
        return objpoints, imgpoints

    def _calibrate(objpoints, imgpoints):
        '''Do camera calibration given object points and image points'''
        img_size = (Camera.img.shape[1], Camera.img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                           img_size, None, None)
        return mtx, dist

    def _saveCalibration(mtx, dist):
        '''Save the camera calibration result for later use'''
        dist_pickle = {}
        dist_pickle[Camera.mtx_string] = mtx
        dist_pickle[Camera.dist_string] = dist
        pickle.dump(dist_pickle, open(Camera.calibration_archive, "wb"))

    def _undistortTestImage(mtx, dist):
        '''Save the undistorted test image'''
        dst = cv2.undistort(Camera.img, mtx, dist, None, mtx)
        cv2.imwrite(Camera.test_undistort_image_path, dst)
        Plotting.plotResult(Camera.img, dst)
        return dst

    #-------------
    # Camera API
    #-------------

    def calibrate():
        '''Call this function to calibrate your camera'''
        
        # Set up and collect the calibration points
        objp = Camera._prepareObjects()
        objpoints, imgpoints = Camera._collectObjectPoints(objp)
        
        # Calibrate the camera and save the calibration matrix in file
        mtx, dist = Camera._calibrate(objpoints, imgpoints)
        Camera._saveCalibration(mtx, dist)
        
        # Undistort the test image and display it in comparisson
        dst = Camera._undistortTestImage(mtx, dist)
        Plotting.plotUndistortedImage(Camera.img, dst)

    def getCalibrationData():
        '''Get the calibration data from the archive'''
        with open(Camera.calibration_archive, "rb") as f:
            pickle_data = pickle.load(f)
            mtx = pickle_data[Camera.mtx_string]
            dist = pickle_data[Camera.dist_string]
        return mtx, dist

    def undistort(image, mtx, dist):
        '''Undistort an image'''
        return cv2.undistort(image, mtx, dist, None, mtx)
