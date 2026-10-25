import os
import numpy as np
import cv2 as cv

# Video to process and where to save the result
video_path = 'test_videos/challenge02.mp4'
output_path = 'output_videos/challenge02.mp4'

calibration = np.load('camera_cal/calib.npz')
mtx = calibration['mtx']
dist = calibration['dist']

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

os.makedirs('output_videos', exist_ok=True)
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

newCameraMtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

if w == 1280:
    src = np.float32([[600, 460], [335, 660], [998, 660], [712, 460]])
    dst = np.float32([[300, 0], [300, 720], [980, 720], [980, 0]])
else:
    src = np.float32([[431, 340], [148, 540], [849, 540], [534, 340]])
    dst = np.float32([[225, 0], [225, 540], [735, 540], [735, 0]])
M = cv.getPerspectiveTransform(src, dst)
Minv = cv.getPerspectiveTransform(dst, src)
left = int(dst[1][0])
right = int(dst[2][0])

xm_per_pix = 3.7 / (right - left)
ym_per_pix = 30 / h
car_center = left + (w / 2 - src[1][0]) * (right - left) / (src[2][0] - src[1][0])

left_fits = []
right_fits = []

while True:
    ret, frame = cap.read()
    if not ret: break

    # Step 2: undistort
    img = frame
    if w == 1280:
        img = cv.undistort(frame, mtx, dist, None, newCameraMtx)

    # Step 3: binary image
    b, g, r = cv.split(img)
    white = cv.adaptiveThreshold(r, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, -35)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    yellow = cv.inRange(hsv, np.array([15, 80, 100]), np.array([30, 255, 255]))
    binary = cv.medianBlur(cv.bitwise_or(white, yellow), 3)

    # Step 4: bird's eye view
    warped = cv.warpPerspective(binary, M, (w, h), flags=cv.INTER_NEAREST)

    # Step 5: histogram and sliding windows
    histogram = np.sum(warped[h // 2:, :], axis=0)
    leftx_current = np.argmax(histogram[left // 2:w // 2]) + left // 2
    rightx_current = np.argmax(histogram[w // 2:(w + right) // 2]) + w // 2

    nonzeroy, nonzerox = warped.nonzero()
    window_height = h // 9
    left_inds = []
    right_inds = []
    left_windows = 0
    right_windows = 0

    for window in range(9):
        y_low = h - (window + 1) * window_height
        y_high = h - window * window_height
        in_window = (nonzeroy >= y_low) & (nonzeroy < y_high)

        good_left = (in_window & (nonzerox >= leftx_current - 100) & (nonzerox < leftx_current + 100)).nonzero()[0]
        good_right = (in_window & (nonzerox >= rightx_current - 100) & (nonzerox < rightx_current + 100)).nonzero()[0]
        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left) > 50:
            leftx_current = int(np.mean(nonzerox[good_left]))
            left_windows += 1
        if len(good_right) > 50:
            rightx_current = int(np.mean(nonzerox[good_right]))
            right_windows += 1

    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)

    left_fit = left_fits[-1] if left_fits else None
    right_fit = right_fits[-1] if right_fits else None
    if len(left_inds) > 500 and left_windows >= 3:
        left_fit = np.polyfit(nonzeroy[left_inds], nonzerox[left_inds], 2)
    if len(right_inds) > 500 and right_windows >= 3:
        right_fit = np.polyfit(nonzeroy[right_inds], nonzerox[right_inds], 2)

    # Lane width check
    if left_fit is not None and right_fit is not None:
        width_bottom = (np.polyval(right_fit, h - 1) - np.polyval(left_fit, h - 1)) * xm_per_pix
        width_top = (np.polyval(right_fit, 0) - np.polyval(left_fit, 0)) * xm_per_pix
        if 2 < width_bottom < 5.5 and 2 < width_top < 5.5:
            left_fits = (left_fits + [left_fit])[-5:]
            right_fits = (right_fits + [right_fit])[-5:]

    if len(left_fits) == 0 or len(right_fits) == 0:
        result = img
    else:
        left_fit = np.mean(left_fits, axis=0)
        right_fit = np.mean(right_fits, axis=0)

        # Step 6: curvature and position
        y_eval = (h - 1) * ym_per_pix
        radius = []
        for fit in (left_fit, right_fit):
            A = fit[0] * xm_per_pix / ym_per_pix ** 2
            B = fit[1] * xm_per_pix / ym_per_pix
            radius.append((1 + (2 * A * y_eval + B) ** 2) ** 1.5 / abs(2 * A))
        curvature = (radius[0] + radius[1]) / 2

        ploty = np.arange(h)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        offset = (car_center - (left_fitx[-1] + right_fitx[-1]) / 2) * xm_per_pix

        # Step 7: draw the lane
        lane = np.zeros_like(img)
        for y in range(h):
            cv.line(lane, (int(left_fitx[y]), y), (int(right_fitx[y]), y), (0, 255, 0), 1)
        for y in range(h - 1):
            cv.line(lane, (int(left_fitx[y]), y), (int(left_fitx[y + 1]), y + 1), (0, 0, 255), 20)
            cv.line(lane, (int(right_fitx[y]), y), (int(right_fitx[y + 1]), y + 1), (255, 0, 0), 20)
        lane = cv.warpPerspective(lane, Minv, (w, h), flags=cv.INTER_NEAREST)
        result = cv.addWeighted(img, 1, lane, 0.4, 0)

        # Step 8: text
        side = 'right' if offset > 0 else 'left'
        cv.putText(result, 'Radius of Curvature = %d(m)' % curvature, (40, 60), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv.putText(result, 'Vehicle is %.2fm %s of center' % (abs(offset), side), (40, 120), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    out.write(result)
    cv.imshow('Lane detection', result)
    if (cv.waitKey(1) & 0xff) == 27: break

cap.release()
out.release()
cv.destroyAllWindows()
