import numpy as np
import cv2 as cv

binary = cv.imread('output_images/binary.png', 0)

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_NEAREST)  # keep same size as input image

    return warped

# Trapezoid on the road and the rectangle it is mapped to
src = np.float32([[600, 460], [335, 660], [998, 660], [712, 460]])
dst = np.float32([[300, 0], [300, 720], [980, 720], [980, 0]])

warped_binary = warper(binary, src, dst)
cv.imwrite('output_images/warped_binary.png', warped_binary)

# Same transform on the color image, with the source and destination points drawn
img = cv.imread('output_images/undistorted.png')
warped = warper(img, src, dst)
for i in range(4):
    x1, y1 = src[i]
    x2, y2 = src[(i + 1) % 4]
    cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
    x1, y1 = dst[i]
    x2, y2 = dst[(i + 1) % 4]
    cv.line(warped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

cv.imshow('Source and destination points', cv.resize(np.hstack((img, warped)), None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR))
cv.imshow('Warped binary', warped_binary)

cv.waitKey(0)
cv.destroyAllWindows()
