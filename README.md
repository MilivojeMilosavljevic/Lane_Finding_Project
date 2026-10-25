**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./docs/calibration_corners.png "Chessboard corners"
[image1]: ./docs/undistort_calibration.png "Undistorted"
[image2]: ./docs/undistort_road.png "Road Transformed"
[image3]: ./docs/binary_combo.png "Binary Example"
[image4]: ./docs/warped_straight_lines.png "Warp Example"
[image5]: ./docs/color_fit_lines.png "Fit Visual"
[image6]: ./docs/example_output.png "Output"
[video1]: ./output_videos/project_video03_h264.mp4 "Video"

### Sources

https://docs.opencv.org/4.x/

Lab exercise scripts (dosiv-py)

Presentation "03 Detekcija ivica i segmentacija.pdf", slides 38–41

Presentation "10 Algoritmi u AV.pdf", slides 26–35

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

I split the code by steps: 01_calibration.py, 02_undistort.py, 03_threshold.py, 04_perspective.py, 05_lane_lines.py, 06_draw_lane.py and 07_video.py. Each step saves its result to output_images/ (the calibration to camera_cal/calib.npz) and the next one loads it, so I could run and check every step on its own. The test image is chosen in 03_threshold.py.

The test images and videos come from two cameras, and the code picks the camera by the size of the image. The 1280x720 ones (test*.jpg, straight_lines*.jpg, project_video03 and challenge01–03) are from the camera that the chessboard images belong to. The 960x540 ones (solidWhite*.jpg, solidYellow*.jpg, whiteCarLaneSwitch.jpg, project_video01 and project_video02) are from a different camera that has no chessboard images, so they are not undistorted and they get their own perspective points. The correction is not needed for this camera anyway, because on its straight road images the lane lines are already straight. The three challange*.jpg images are also 960x540, so they go through the second camera, although they are smaller copies of frames from challenge01 (more about this in the discussion).

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In 01_calibration.py I build the ideal chessboard as a grid of points with z = 0, since the board is flat. The board has 10x7 squares, but findChessboardCorners looks for the inner corners where four squares meet, so the grid is 9x6. I first tried 10x7 and nothing was found. Each image is converted to grayscale, and when the corners are found they are refined with cornerSubPix and stored next to the ideal points. calibrateCamera turns these pairs into the camera matrix (mtx) and the distortion coefficients (dist), which are saved to camera_cal/calib.npz. The rotation and translation vectors are only used to project the board back and check the error.

The board was found on 17 of the 20 images. On the other three (calibration1, calibration4 and calibration5) a part of the board is cut off by the edge of the image, so the whole 9x6 pattern is not visible, and findChessboardCorners only reports the board when it finds all the corners it was asked for.

The corners found on calibration2.jpg, connected row by row:

![alt text][image0]

02_undistort.py computes a new camera matrix with getOptimalNewCameraMatrix and corrects the image with undistort. With alpha = 1 every original pixel stays in the image and curved black borders appear around it, with alpha = 0 the image is cropped so only valid pixels remain. I started with 1, but the black borders later showed up as lines in the binary image, so I switched to 0.

calibration1.jpg before and after the correction. On the left the rows of squares are bent, on the right they are straight:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

test1.jpg before and after the correction:

![alt text][image2]

The correction leaves the center of the image as it is and moves pixels more the further they are from it. Without it the lane would look narrower than it really is.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My first try was a Sobel gradient in x, but on the test images it mostly found the edges of trees and the barrier. The 10th presentation shows a second way, where the lines are found by color, so I went with that. The binary image is made in 03_threshold.py from two masks:

* white: the red channel, since white lines are bright in all channels
* yellow: HSV with H 15–30, S ≥ 80 and V ≥ 100

The masks are joined with OR, and a 3x3 median blur removes single dots. A 5x5 median was too strong, it erased the far dashes, which are only a few pixels wide.

For the white mask I started with a fixed threshold on the red channel. At 200 the light concrete on test1.jpg and test4.jpg passed as white, so I raised it to 220. That fixed the concrete, but the dashed lines were left with very few pixels, and in challenge02 the faded dashes disappeared completely. One value could not work for both. The 3rd presentation (slides 38–41) says that a fixed threshold works well only when the lighting is the same everywhere, and shows adaptive thresholding for the other cases, so in the end I used an adaptive threshold, with the same function as in the lab script 03_09_adaptive_thrashold.py: a pixel is white if it is brighter than the mean of its 25x25 neighbourhood by more than 35. The downside is that it also catches other bright edges (the barrier, cars, shadow edges), but those are mostly off the road and get removed in the next steps.

The yellow range first went up to H = 35. In challenge03 the sunny leaves next to the road are yellow-green and they ended up in the yellow mask, so the lines were pulled into the bushes. The yellow lines on the road have a hue of about 20 and the leaves are mostly above 30, so I lowered the upper limit to 30. Most of the leaves were gone and the lines stayed the same.

The binary image of test6.jpg. Both lane lines are clear, and the noise from the trees and the cars is outside the part of the road that is used in the next step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This is in 04_perspective.py. I used the warper() function from examples/example.py, which maps a trapezoid on the road to a rectangle with getPerspectiveTransform and warpPerspective. The result is the road seen from above (bird's eye view), where the lane lines are vertical and parallel. Each camera has its own points:

| Point | 1280x720 source | 1280x720 destination | 960x540 source | 960x540 destination |
|:---:|:---:|:---:|:---:|:---:|
| top left | 600, 460 | 300, 0 | 431, 340 | 225, 0 |
| bottom left | 335, 660 | 300, 720 | 148, 540 | 225, 540 |
| bottom right | 998, 660 | 980, 720 | 849, 540 | 735, 540 |
| top right | 712, 460 | 980, 0 | 534, 340 | 735, 0 |

I picked the points on images of a straight road (straight_lines1 and straight_lines2 for the first camera, solidWhiteRight and solidYellowLeft for the second). The road is straight there, so from above both lines have to be vertical and parallel, and I moved the points until the lines landed on the destination x. My first trapezoid for the first camera reached down to y = 690 and caught the hood of the car, so I raised it to 660. Above y = 460 the road is so far away that it turns into a blur. When I switched to alpha = 0 I had to pick the points again, because the corrected image is scaled differently.

I tried both orders, binary image first and bird's eye view first, and got almost the same result, so I kept the order from the task.

straight_lines1.jpg with the source points drawn, and the warped result with the destination points drawn. The lines are vertical and lie on x = 300 and x = 980:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is the first part of 05_lane_lines.py, done after slides 32–34 of the 10th presentation. I sum the white pixels of every column in the bottom half of the bird's eye view. The two highest peaks, one on each side of the middle, are where the lines start. I only look for the peaks near the place where the lines should be (on the 1280x720 images between x = 150 and x = 1130), because on test1.jpg the right peak first landed on the edge of a white car, which moved the car half a meter too far to the left.

From the peaks the sliding windows go up the image, 9 per line and 200 px wide. Each window takes the white pixels inside it, and if there are more than 50, the next window moves to their mean x. Through the collected pixels of each line I fit

x = A·y² + B·y + C

with np.polyfit. It is x as a function of y, because the lines are nearly vertical. The windows are not always centered on the line: in a curve they are one step behind, and in the gaps of the dashed line they do not move. That is fine, since they only collect pixels and the polynomial goes through the middle of them.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is the second part of 05_lane_lines.py.

To get the radius in meters I needed to know how many meters one pixel is in the bird's eye view. I looked up the values that are usually used for this kind of project: a highway lane is about 3.7 m wide, and the bird's eye view shows about 30 m of road ahead. In my bird's eye view the lines are 680 px apart and the image is 720 px high, so one pixel is 3.7 / 680 m across and 30 / 720 m along the road. For the 960x540 camera it is 3.7 / 510 and 30 / 540.

With these numbers I fit both polynomials again, this time in meters, and calculate the radius at the bottom of the image, which is the part of the road closest to the car:

R = (1 + (2Ay + B)²)^(3/2) / |2A|

I do this for both lines and write their average on the image.

For the position of the car I assumed that the camera is in the middle of the car. At first I took the middle of the image (x = 640) as the middle of the car, but then the car came out left of the center in almost every frame. The camera looks straight ahead, so the middle of the car is at the optical center of the camera, which is the cx value of the camera matrix from the calibration (x = 680 in the undistorted image). The 960x540 camera has no calibration, so for it the middle of the image is used. The middle of the lane is halfway between the two lines at the bottom of the image. The difference between these two, converted to meters, is the offset. If it is positive the car is right of the center, and if it is negative it is left. The example output that came with the task (examples/example_output.jpg) shows 0.17 m left, and on the same frame of the project video I get 0.15 m left.

On straight roads the radius comes out at several kilometers or more, which makes sense for a straight road, and the car is within about 0.1 m of the center. In the curves of the test images the radius is about 1 km (between 920 and 2280 m).

The problem here was that the left and the right line often give a very different radius, for example 1831 m and 382 m on test4.jpg. The dashed line has only 2–3 dashes in the image, so its polynomial is not as reliable as the one of the solid line, and since the radius depends on 1/A, a small difference in A changes the result a lot. That is why I use the average of both lines, and in the video also the average of the last 5 frames.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

06_draw_lane.py fills the space between the two polynomials with green row by row in the bird's eye view (the warped image from point 3), draws the left line red and the right line blue, warps this back with the same warper() and the points swapped, and lays it over the image with addWeighted. The radius and the offset are written with putText.

![alt text][image6]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

project_video03:

https://github.com/user-attachments/assets/da78dbde-03ce-49bc-83b1-6253e7d48998

07_video.py runs steps 2–6 on every frame and picks the camera by the size of the frame. A new polynomial is used only when the line has at least 500 pixels and was found in at least 3 of the 9 windows. The new lines are also kept only if the lane between them is 2–5.5 m wide at the bottom and at the top, otherwise the lines from the previous frames are used. The lane is drawn with the average of the polynomials from the last 5 frames, so it stays calm. The script saves an mp4v file, which I converted to H.264 with ffmpeg so it plays on GitHub.

The lane is found well on all three project videos (project_video03 above, which is from the first camera, and project_video01 and project_video02 from the second camera) and on challenge01. The lane stays calm and only wobbles a little in a few places. The results of all six videos are in the output_videos folder (project_video01_h264.mp4, project_video02_h264.mp4, project_video03_h264.mp4, challenge01_h264.mp4, challenge02_h264.mp4 and challenge03_h264.mp4).

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of my time went into the numbers: the white threshold (first fixed, then adaptive), the yellow range, the median size and the trapezoid points. A value that fixed one image often broke another, so after every change I checked all test images and all videos again.

The pipeline is weakest when the road is not flat or the lines are hard to see. In project_video03 the car goes over a few bumps where it drives onto and off the concrete bridge. The camera tilts for a moment, the fixed trapezoid no longer matches the road, and the lane moves a little for a couple of frames. Shadows, worn lines, glare, night or rain would leave too few line pixels, and a lane change would confuse the histogram. To make it more robust I would search around the previous polynomial instead of starting from zero every frame, and place the trapezoid automatically instead of using fixed points.

challenge02 and challenge03 are much harder, and I am not sure how to solve them properly with this approach:

* challenge02: the white dashes are worn out on light concrete, the yellow line fades in the distance and gets mixed up with the concrete wall next to it, it is dark under the bridge, and the camera looks at the road at a slightly different angle, so my trapezoid does not fit well. The lane is drawn, but it is not always stable.
* challenge03: a mountain road with very sharp curves where the lines leave the trapezoid, yellow-green leaves next to the road, sun glare and deep shadows. At first nothing worked. Lowering the yellow hue limit (to leave the leaves out), using a line only if it is found in at least 3 of the 9 windows, and keeping new lines only if the lane between them has a normal width made it usable for most of the video, but it still fails in the sharpest curves near the end.

The three challange*.jpg images are smaller copies of frames from challenge01, but because of their size the code treats them as the second camera. The lane is still found, but the result is less accurate (no undistortion and the trapezoid reaches over the hood).
