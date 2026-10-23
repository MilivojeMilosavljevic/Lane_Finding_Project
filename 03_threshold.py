import numpy as np
import cv2 as cv

# Load calibrated camera parameters
calibration = np.load('camera_cal/calib.npz')
mtx = calibration['mtx']
dist = calibration['dist']

# Load one of the test images and undistort it (as in 09_02_undistort.py)
img = cv.imread('test_images/test1.jpg')
h, w = img.shape[:2]
newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
img = cv.undistort(img, mtx, dist, None, newCameraMtx)
cv.imshow('Undistorted image', img)

# White lines: threshold on the red channel (split as in 00_03_colors.py, threshold as in 03_08_manual_thrashold.py)
b, g, r = cv.split(img)
ret, white = cv.threshold(r, 220, 255, cv.THRESH_BINARY)
cv.imshow('White (red channel)', white)

# Yellow lines: color masking in HSV (as in 06_01_color_masking.py)
lower_yellow = np.array([15, 80, 100])
upper_yellow = np.array([35, 255, 255])

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
cv.imshow('Yellow (HSV)', yellow)

# Binary image: white or yellow, then remove the noise (median blur as in 02_02_low_pass.py)
binary = cv.bitwise_or(white, yellow)
binary = cv.medianBlur(binary, 3)
cv.imshow('Binary', binary)

# Save the binary image for the next step (04_perspective.py), as PNG so it stays exactly black and white
cv.imwrite('output_images/binary.png', binary)

cv.waitKey(0)
cv.destroyAllWindows()
