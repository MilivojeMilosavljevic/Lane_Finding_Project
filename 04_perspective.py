import numpy as np
import cv2 as cv

# Load the binary image made by 03_threshold.py (as grayscale, as in 03_07_gradient.py)
binary = cv.imread('output_images/binary.png', 0)

# Perspective transform (warper() from examples/example.py)
def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_NEAREST)  # keep same size as input image

    return warped

# Trapezoid on the road (top left, bottom left, bottom right, top right),
# chosen on straight_lines1.jpg and straight_lines2.jpg so that straight lines become vertical
src = np.float32([[600, 460], [335, 660], [998, 660], [712, 460]])

# Rectangle the trapezoid is mapped to
dst = np.float32([[300, 0], [300, 720], [980, 720], [980, 0]])

warped_binary = warper(binary, src, dst)

# Draw the trapezoid on the binary image (as cv.line in 04_01_hough_lines.py)
marked = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
for i in range(4):
    x1, y1 = src[i]
    x2, y2 = src[(i + 1) % 4]
    cv.line(marked, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

cv.imshow('Binary image', marked)
cv.imshow('Warped binary', warped_binary)

cv.waitKey(0)
cv.destroyAllWindows()
