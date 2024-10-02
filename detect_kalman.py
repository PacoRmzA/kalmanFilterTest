import cv2 as cv
import numpy as np
from math import isnan

cv.namedWindow("frame")
cv.namedWindow("detect")
cap = cv.VideoCapture(0)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

k = cv.KalmanFilter(4,2) # 4 state vars (pos xy, vel xy), 2 measurement vars (pos xy)
k.measurementMatrix = np.array( # H matrix, only positions are directly measurable
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)

k.transitionMatrix = np.array( # F matrix, the off-diagonal ones correspond to velocity influence on position
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)

k.processNoiseCov = np.array( # Q matrix, set to a small value with only individual variable variance (diagonal) for testing
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03

k.measurementNoiseCov = np.array( # R matrix, set to a larger value with only individual variable variance (diagonal) for testing
            [[5, 0],
             [0, 5]], np.float32)

init = False

while True:
    
    ret, frame = cap.read()
    if frame is None:
        break
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (63, 164, 48), (121, 255, 94)) # hardcoded HSV values for green alarm clock test
    cv.erode(frame_threshold, kernel, frame_threshold, iterations=2)
    cv.dilate(frame_threshold, kernel, frame_threshold, iterations=4)

    np_seg = np.array(frame_threshold)
    segmentation = np.where(np_seg == 255)
    cx = np.mean(segmentation[1]) 
    cy = np.mean(segmentation[0]) 

    if not isnan(cx) and not isnan(cy):
        cv.circle(frame,(int(cx),int(cy)), 5, (0,0,255), -1) # red circle marking raw centroid of masked pixels
        if not init:
            k.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            k.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            init = True
        k.correct(np.array([cx, cy], np.float32))
    
    if init:
        pred_arr = k.predict()
        cv.circle(frame,(int(pred_arr[0]),int(pred_arr[1])), 5, (0,255,0), -1)
    else:
        continue
    
    cv.imshow("frame", frame)
    cv.imshow("detect", frame_threshold)
    
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break