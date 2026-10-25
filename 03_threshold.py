import numpy as np
import cv2 as cv

# Load calibrated camera parameters
calibration = np.load('camera_cal/calib.npz')
mtx = calibration['mtx']
dist = calibration['dist']

# Load one of the test images
img = cv.imread('test_images/test5.jpg')
h, w = img.shape[:2]

# Obtain the new camera matrix and undistort the image
if w == 1280:
    newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    img = cv.undistort(img, mtx, dist, None, newCameraMtx)
cv.imshow('Undistorted image', img)
cv.imwrite('output_images/undistorted.png', img)

# White lines
b, g, r = cv.split(img)
white = cv.adaptiveThreshold(r, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, -35)
cv.imshow('White (red channel)', white)

# Yellow lines
lower_yellow = np.array([15, 80, 100])
upper_yellow = np.array([30, 255, 255])
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
cv.imshow('Yellow (HSV)', yellow)

# Binary image
binary = cv.bitwise_or(white, yellow)
binary = cv.medianBlur(binary, 3)
cv.imshow('Binary', binary)
cv.imwrite('output_images/binary.png', binary)

cv.waitKey(0)
cv.destroyAllWindows()
