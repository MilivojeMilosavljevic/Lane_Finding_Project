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

[image1]: ./docs/undistort_calibration.png "Undistorted"
[image2]: ./docs/undistort_road.png "Road Transformed"
[image3]: ./docs/binary_combo.png "Binary Example"
[image4]: ./docs/warped_straight_lines.png "Warp Example"
[image5]: ./docs/color_fit_lines.png "Fit Visual"
[image6]: ./docs/example_output.png "Output"
[video1]: ./output_videos/project_video03_h264.mp4 "Video"

### Sources

https://docs.opencv.org/4.x/

Lab exercise scripts (`dosiv-py`)

Presentation "10 Algoritmi u AV.pdf", slides 26–35

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

I split the code by steps: `01_calibration.py`, `02_undistort.py`, `03_threshold.py`, `04_perspective.py`, `05_lane_lines.py`, `06_draw_lane.py` and `07_video.py`. Each step saves its result to `output_images/` (the calibration to `camera_cal/calib.npz`) and the next one loads it, so I could run and check every step on its own. The test image is chosen in `03_threshold.py`.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In `01_calibration.py` I build the ideal chessboard as a 9x6 grid of points with z = 0, since the board is flat. I first tried 10x7, because that is how many squares the board has, but nothing was found: `findChessboardCorners` looks for the inner corners where four squares meet. Each image is converted to grayscale, and when all corners are found they are refined with `cornerSubPix` and stored next to the ideal points. `calibrateCamera` turns these pairs into the camera matrix `mtx` and the distortion coefficients `dist`, which go to `camera_cal/calib.npz`. The rotation and translation vectors are only used to project the board back and check the error.

The board was found on 17 of the 20 images, on the other three it is cut off by the edge of the image. A projected corner misses the real one by 0.72 px on average, and `k1 = -0.247` confirms the barrel distortion of this lens.

`02_undistort.py` computes a new camera matrix with `getOptimalNewCameraMatrix` and corrects the image with `undistort`. With `alpha = 1` every original pixel stays in the image and curved black borders appear around it, with `alpha = 0` the image is cropped so only valid pixels remain. I started with 1, but the black borders later showed up as lines in the binary image, so I switched to 0.

`calibration1.jpg` before and after the correction. On the left the rows of squares are bent, on the right they are straight:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

`test1.jpg` before and after the correction:

![alt text][image2]

The correction leaves the center of the image as it is and moves pixels more the further they are from it: about 23 px at the bottom of the lane lines and up to 155 px in the corners. Without it the lane would look narrower than it really is.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My first try was a Sobel gradient in x, but on the test images it mostly found the edges of trees and the barrier. The 10th presentation shows a second way, where the lines are found by color, so I went with that. The binary image is made in `03_threshold.py` from two masks:

* white: the red channel, since white lines are bright in all channels
* yellow: HSV with H 15–35, S ≥ 80 and V ≥ 100

The masks are joined with OR, and a 3x3 median blur removes single dots. A 5x5 median was too strong, it erased the far dashes, which are only a few pixels wide.

For the white mask I started with a fixed threshold on the red channel. At 200 the light concrete on `test1.jpg` and `test4.jpg` passed as white, so I raised it to 220. That fixed the concrete, but the dashed lines were left with very few pixels, and in `challenge02` the faded dashes disappeared completely. One value could not work for both, so in the end I used an adaptive threshold (`cv.adaptiveThreshold`): a pixel is white if it is brighter than the mean of its 25x25 neighbourhood by more than 35. The dashed line on `straight_lines1` went from 2432 to 7389 pixels. The downside is that it also catches other bright edges (the barrier, cars, shadow edges), but those are mostly off the road and get removed in the next steps.

The binary image of `test6.jpg`. Both lane lines are clear, and the noise from the trees and the cars is outside the part of the road that is used in the next step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This is in `04_perspective.py`. I used the `warper()` function from `examples/example.py`, which maps a trapezoid on the road to a rectangle with `getPerspectiveTransform` and `warpPerspective`:

```python
src = np.float32([[600, 460], [335, 660], [998, 660], [712, 460]])
dst = np.float32([[300, 0], [300, 720], [980, 720], [980, 0]])
```

I picked the points on the two `straight_lines` images. The road is straight there, so from above both lines have to be vertical and parallel, and I moved the points until they landed on x = 300 and x = 980. My first trapezoid reached down to y = 690 and caught the hood of the car, so I raised it to 660. Above y = 460 the road is so far away that it turns into a blur. When I switched to `alpha = 0` I had to pick the points again, because the undistorted image is scaled differently.

I tried both orders, binary image first and bird's eye view first, and got almost the same result, so I kept the order from the task. These points only fit the 1280x720 camera. The 960x540 images and videos come from a different camera with no chessboard images, so I left them out.

`straight_lines1.jpg` with the source points drawn, and the warped result with the destination points drawn. The lines are vertical and lie on x = 300 and x = 980:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is the first part of `05_lane_lines.py`, done after slides 32–34 of the 10th presentation. I sum the white pixels of every column in the bottom half of the bird's eye view. The two highest peaks, one on each side of the middle, are where the lines start. I only look for the peaks between x = 150 and x = 1130, because on `test1.jpg` the right peak first landed on the edge of a white car at x = 1232, which put the car 0.84 m left of the center instead of 0.32 m.

From the peaks the sliding windows go up the image, 9 per line and 200 px wide. Each window takes the white pixels inside it, and if there are more than 50, the next window moves to their mean x. Through the collected pixels of each line I fit

x = A·y² + B·y + C

with `np.polyfit`. It is x as a function of y, because the lines are nearly vertical. The windows are not always centered on the line: in a curve they are one step behind, and in the gaps of the dashed line they do not move. That is fine, since they only collect pixels and the polynomial goes through the middle of them.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is the second part of `05_lane_lines.py`. For x I used the US lane width of 3.7 m, which is 680 px in the bird's eye view. For y I did not want to guess, so I measured the dashed lines on the straight road images: a dash is 3.05 m long and about 107 px in the image, and a dash with its gap is 12.2 m and about 383 px, which gives about 0.03 m per pixel. Both polynomials are fitted again in meters, and the radius is computed at the bottom of the image with

R = (1 + (2Ay + B)²)^(3/2) / |2A|

The average of the left and the right line is shown on the image. For the position I assume the camera is in the middle of the car, so the image center (x = 640, which is x ≈ 613 in the bird's eye view) is the car, and the middle between the two lines at the bottom is the lane center. Their difference in meters is the offset, positive to the right.

On the straight road images the radius is 7–9 km and the car is 0.13–0.15 m left of the center, close to the 0.17 m in the example output that came with the task. In the curves of the test images the average radius is between about 480 and 1180 m.

The left and the right radius are often far apart (on `test4.jpg` 949 m and 198 m). The dashed line has only 2–3 dashes in the image and 4–5 times fewer pixels than the solid one, so its polynomial is less certain, and because the radius depends on 1/A a small error in A becomes a big one. The curve is also judged from only about 22 m of road, so the radius is the least reliable number in the project.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`06_draw_lane.py` fills the space between the two polynomials with green row by row in the bird's eye view, draws the left line red and the right line blue, warps this back with the same `warper()` and the points swapped, and lays it over the image with `addWeighted`. The radius and the offset are written with `putText`.

![alt text][image6]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[project_video03_h264.mp4](./output_videos/project_video03_h264.mp4)

`07_video.py` runs steps 2–6 on every frame. A new polynomial is used only when the line has at least 500 pixels, and the lane is drawn with the average of the polynomials from the last 5 frames, so it stays calm. The script saves an `mp4v` file, which I converted to H.264 with ffmpeg so it plays on GitHub.

The lane is found on all 1260 frames, the car stays between 0.05 and 0.63 m left of the center, and the position never jumps more than 0.06 m between two frames. It also works on `challenge01`: [challenge01_h264.mp4](./output_videos/challenge01_h264.mp4).

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of my time went into the numbers: the white threshold (200, then 220, then adaptive), the yellow range, the median size, the trapezoid points and the histogram limits. A value that fixed one image often broke another, so after every change I checked all test images and the video again.

The pipeline is weakest when the road is not flat or the lines are hard to see. When the car bounces over a bump, the camera tilts and the fixed trapezoid stops matching the road for a moment, so the lane shifts slightly for a couple of frames. It is only a small shift and the averaging hides most of it, but on hills it would be worse. Shadows, worn lines, cars over the lines, night or rain would leave the windows without enough pixels, and the histogram expects one line on each side of the image, which is not true during a lane change. To make it more robust I would compute the curvature mostly from the solid line, search around the previous polynomial instead of starting the windows from zero every frame, throw away polynomials whose lines are not parallel, and find the vanishing point to place the trapezoid automatically.

**challenge02 – the hardest example**

[challenge02_h264.mp4](./output_videos/challenge02_h264.mp4)

I did not manage to make this video work well, but it shows most of the weak points at once. The lane is drawn on every frame, but the position sometimes jumps up to 0.72 m between two frames. What I found:

* the white dashes are faded on light concrete (red value about 200–235, the road about 150). With the fixed threshold of 220 no lane was drawn in the first 4 seconds; the adaptive threshold fixed that.
* the left line gets mixed up with the small concrete wall next to it. Far away the yellow line becomes thin and pale (its saturation drops from about 140 to 57, below the 80 the mask needs), so the upper windows have no yellow pixels. The bright bottom edge of the wall runs through them, the adaptive threshold takes it for a white line, and the polynomial gets pulled to the left. The small yellow markers on top of the wall are caught by the yellow mask too, but they fall outside the bird's eye view.
* under the bridge (around second 4) everything is dark, the yellow line does not pass the HSV threshold and the edge of the shadow becomes a white blob. Right after it the right line runs off to the right, and the car comes out 2.10 m left of the center.
* the camera looks at the road at a different angle than in the project video (the vanishing point is at y ≈ 443 instead of 420), so my trapezoid does not fit and the lines from above are not parallel.

I tried a separate trapezoid just for this video, which helped a lot, but tuning points by hand for one video is not a real fix. I also tried throwing away non-parallel polynomials, but because of the wrong trapezoid most frames were thrown away and the lane froze for seconds. Using only yellow pixels for the left line ignored the wall and cut the frames with a badly bent left line from 110 to 5, but it only works where the left line is yellow (on `straight_lines2` it is white).

The other challenge video, `challenge03`, is a mountain road with sharp curves, a double yellow line and sun glare, and my pipeline does not work on it at all.
