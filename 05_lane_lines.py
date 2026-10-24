import numpy as np
import cv2 as cv

binary_warped = cv.imread('output_images/warped_binary.png', 0)
h, w = binary_warped.shape
out_img = cv.cvtColor(binary_warped, cv.COLOR_GRAY2BGR)

# Histogram of the bottom half, the peaks are where the lines start (searched only between x = 150 and 1130)
histogram = np.sum(binary_warped[h // 2:, :], axis=0)
midpoint = w // 2
leftx_base = np.argmax(histogram[150:midpoint]) + 150
rightx_base = np.argmax(histogram[midpoint:1130]) + midpoint

# Sliding windows
nwindows = 9
margin = 100
minpix = 50
window_height = h // nwindows

nonzeroy, nonzerox = binary_warped.nonzero()

leftx_current = leftx_base
rightx_current = rightx_base
left_lane_inds = []
right_lane_inds = []

for window in range(nwindows):
    win_y_low = h - (window + 1) * window_height
    win_y_high = h - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = int(np.mean(nonzerox[good_right_inds]))

left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Polynomial x = A*y^2 + B*y + C for each line
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

out_img[lefty, leftx] = [0, 0, 255]
out_img[righty, rightx] = [255, 0, 0]

ploty = np.arange(h)
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
for i in range(h - 1):
    cv.line(out_img, (int(left_fitx[i]), i), (int(left_fitx[i + 1]), i + 1), (0, 255, 255), 3)
    cv.line(out_img, (int(right_fitx[i]), i), (int(right_fitx[i + 1]), i + 1), (0, 255, 255), 3)

# Meters per pixel: lane width 3.7 m is 680 px, y measured on the dashed lines (about 0.03 m per pixel)
xm_per_pix = 3.7 / 680
ym_per_pix = 0.03

# Radius of curvature at the bottom of the image, from the polynomials fitted in meters
left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

y_eval = (h - 1) * ym_per_pix
left_curverad = (1 + (2 * left_fit_m[0] * y_eval + left_fit_m[1]) ** 2) ** 1.5 / abs(2 * left_fit_m[0])
right_curverad = (1 + (2 * right_fit_m[0] * y_eval + right_fit_m[1]) ** 2) ** 1.5 / abs(2 * right_fit_m[0])
curvature = (left_curverad + right_curverad) / 2

# Vehicle position, the camera is in the middle of the car (x = 640 is x = 613 in the bird's eye view)
car_center = 300 + (640 - 335) * (980 - 300) / (998 - 335)
left_bottom = left_fit[0] * (h - 1) ** 2 + left_fit[1] * (h - 1) + left_fit[2]
right_bottom = right_fit[0] * (h - 1) ** 2 + right_fit[1] * (h - 1) + right_fit[2]
lane_center = (left_bottom + right_bottom) / 2
offset = (car_center - lane_center) * xm_per_pix

print("Left line:  x = %.6f*y^2 + %.4f*y + %.1f" % tuple(left_fit))
print("Right line: x = %.6f*y^2 + %.4f*y + %.1f" % tuple(right_fit))
print("Radius of curvature: left %.0f m, right %.0f m, mean %.0f m" % (left_curverad, right_curverad, curvature))
print("Vehicle is %.2f m %s of center" % (abs(offset), "right" if offset > 0 else "left"))

np.savez('output_images/lane_fit.npz', left_fit=left_fit, right_fit=right_fit, curvature=curvature, offset=offset)

cv.imshow('Histogram peaks: %d, %d' % (leftx_base, rightx_base), out_img)

cv.waitKey(0)
cv.destroyAllWindows()
