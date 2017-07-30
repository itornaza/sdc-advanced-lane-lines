## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The rubric for this project can be found [here](https://review.udacity.com/#!/rubrics/571/view).

[//]: # (Image References)

[image1]: ./output_images/test_undistort_image_result_1.png "Undistorted chessboard"
[image2]: ./output_images/test_undistort_image_result_2.png "Undistorted road image"
[image3]: ./output_images/binary_mask_result.png "Binary mask example"
[image4]: ./output_images/perspective_result.png "Warp example"
[image5]: ./output_images/sliding_window_result.png "Sliding window visual"
[image6]: ./output_images/fitted_curves_result.png "Fitted curves result"
[image7]: ./output_images/final_result.png "Final result"

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this is contained in a separate class called Camera and is located in its own file "./classes/camera.py".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z = 0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates which is initialized in the `_prepareObject()` method of the `Camera` class, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The processing of the `objpoints` is performed in the `_collectObjectPoints()` internal method of the `Camera` class.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  This is done in the `_calibrate()` internal method of the `Camera` class.

The `Camera` class provides a `calibration()` method to calibrate the camera from scratch and a `getCalibrationData()` method to retrieve the stored calibration data. To run the camera calibration from the command line type: `python main.py -c`

I applied the distortion correction after calibrating the camera to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Exploratory Pipeline for single images

To run the exploratory single image pipeline from the command line type: `python main.py -e`

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The R channel from the RGB colorspace is used for yellow lane detection, the L channel from the HLS colorspace is used for shadows supression and the H channel from the HLS namespace is processed in combination with grayscale combined masks. All of this functionality is included in the `Thresholding` class found in the `./classes/thresholding.py` file. The `lane_detection_mask()` method is the method that handles the binary mask generation. Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the `perspectiveTransform()` method of the `Image_processing` class that can be found in the `./classes/image_processing.py`. The `perspectiveTransform()` method takes as inputs an image (`img`). The source (`src`) and destination (`dst`) points are set up as local variables of this method and are hardcoded for the specific input of the self driving car camera of the project, such as:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 220, 720      | 320, 720      | 
| 1110, 720     | 920, 720      |
| 570, 470      | 320, 0        |
| 720, 470      | 920, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane lines are identified in pixel level in the `slidingWindowInit()` method of the `Image_processing class` found in the `./classes/image_processing.py` file. A sliding window is running through the image and in each window the points of each line are detected and appended to the total line points. After all the points are collected they are fitted into 2nd order polynomials. The `slidingWindowFollowing()` method implements the same technique if the lines from a previous image are already known. This algorithm limits the search within a margin from the known lines in order to identify the new ones in order to expedite processing. The following images show the sliding window processing and the fitted lane lines.

![alt text][image5]
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius is calculated in the `curvature()` method of the `Image_processing` class and the offset of the car from the center of the lane is calculated in the `offsetFromCenter` of the `Image_processing` class. For both calculations the image to real world space is taken into consideration. The curvature is directly computed from the coeficients of the second order polynomials and the offset from center is calculated by locating the lane center from averaging the two lane points at the bottom of the image and comparing it to the center of the frame itself.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the `laneArea()` method of the `Image_processing` class. Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline for video

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [youtube link to my video result](https://youtu.be/kXtFJkXawFY)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 



