import numpy as np
import cv2 as cv

img = cv.imread('output_images/undistorted.png')
lane = np.load('output_images/lane_fit.npz')
left_fit = lane['left_fit']
right_fit = lane['right_fit']
curvature = lane['curvature']
offset = lane['offset']
h, w = img.shape[:2]

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, img_size, flags=cv.INTER_NEAREST)  # keep same size as input image

    return warped

src = np.float32([[600, 460], [335, 660], [998, 660], [712, 460]])
dst = np.float32([[300, 0], [300, 720], [980, 720], [980, 0]])

ploty = np.arange(h)
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

# Lane drawn in the bird's eye view: green area, left line red, right line blue
lane_img = np.zeros((h, w, 3), dtype='uint8')
for y in range(h):
    cv.line(lane_img, (int(left_fitx[y]), y), (int(right_fitx[y]), y), (0, 255, 0), 1)
for y in range(h - 1):
    cv.line(lane_img, (int(left_fitx[y]), y), (int(left_fitx[y + 1]), y + 1), (0, 0, 255), 20)
    cv.line(lane_img, (int(right_fitx[y]), y), (int(right_fitx[y + 1]), y + 1), (255, 0, 0), 20)

# Back to the camera view (src and dst swapped) and over the original image
lane_back = warper(lane_img, dst, src)
result = cv.addWeighted(img, 1, lane_back, 0.4, 0)

side = 'right' if offset > 0 else 'left'
cv.putText(result, 'Radius of Curvature = %d(m)' % curvature, (40, 60), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
cv.putText(result, 'Vehicle is %.2fm %s of center' % (abs(offset), side), (40, 120), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

cv.imshow('Bird\'s eye view lane', lane_img)
cv.imshow('Result', result)

cv.waitKey(0)
cv.destroyAllWindows()
