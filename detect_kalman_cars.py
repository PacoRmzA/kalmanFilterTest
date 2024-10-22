import cv2 as cv
import numpy as np
from math import isnan

# defines matrices necessary for Kalman Filter operation
def defineKalmanFilter():
    k = cv.KalmanFilter(4,2) # 4 state vars (pos xy, vel xy), 2 measurement vars (pos xy)
    k.measurementMatrix = np.array( # H matrix, only positions are directly measurable
                [[1, 0, 0, 0],
                [0, 1, 0, 0]], np.float32)

    # F matrix, the off-diagonal ones correspond to velocity influence on position
    k.transitionMatrix = np.array(
                [[1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], np.float32)

    # Q matrix, set to a small value with only individual variable variance (diagonal) for testing
    k.processNoiseCov = np.array(
                [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], np.float32) * 0.03

    # R matrix, set to a larger value with only individual variable variance (diagonal) for testing
    k.measurementNoiseCov = np.array(
                [[7, 0],
                [0, 7]], np.float32)
    return k

# returns a binary mask filtering all non-yellow pixels as well as detection noise
def yellowThreshold(frame):
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    thresholded = cv.inRange(frame_HSV, (10, 200, 127), (30, 255, 255))
    cv.erode(thresholded, kernel, thresholded, iterations=4)
    cv.dilate(thresholded, kernel, thresholded, iterations=2)
    return thresholded

# returns the average center position of all circles in the frame
def circles(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.blur(gray, (5,5)) # blur improves circle detection
    rows = gray.shape[0]
    c = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=20)
    if c is not None:
        n_circ = 0 # number of circles
        avg_circ_x = 0
        avg_circ_y = 0
        c = np.uint16(np.around(c))
        for i in c[0, :]:
            n_circ += 1
            avg_circ_x += i[0]
            avg_circ_y += i[1]
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(frame, center, radius, (255, 0, 255), 3) # draw magenta circle
        avg_circ_x /= n_circ # avg circle center x position
        avg_circ_y /= n_circ # avg circle center y position
        return [avg_circ_x, avg_circ_y]
    else:
        return None

cap = cv.VideoCapture('videoCarrito.mp4')

# background subtractor, 100 frame history, 300 sensitivity, no shadows
backSub1 = cv.createBackgroundSubtractorMOG2(100, 300, False)
# each call to the subtractor requires an independent instance
backSub2 = cv.createBackgroundSubtractorMOG2(100, 300, False)

# 3px circle kernel for morphological operations
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kf = defineKalmanFilter()

# whether the Kalman Filter has been initialized
init = False
frame_n = 0

if cap.isOpened() == False:
    print("Error opening file")

while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        break

    frame_n += 1

    # fg_color contains non-static yellow pixels (only car wheels, removes motors)
    tr = yellowThreshold(frame)
    fg_color = backSub1.apply(tr)

    n_white = np.sum(fg_color == 255) # number of white pixels
    segmentation = np.where(fg_color == 255) # white pixel positions
    
    # centroid position is the mean position of all white pixels
    cx = np.mean(segmentation[1]) 
    cy = np.mean(segmentation[0])

    # circles are detected from the part of the frame that is not static (only yellow car)
    fg_original = backSub2.apply(frame)
    frame_mov = cv.bitwise_and(frame, cv.cvtColor(fg_original, cv.COLOR_GRAY2BGR))
    avg_circ_center = circles(frame_mov)

    # frames with valid detections
    if n_white >= 60 and n_white < 1000:
        if not init:
            # initialize predictions for Kalman Filter
            kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            init = True
        kf.correct(np.array([cx, cy], np.float32)) # use measurement to correct prediction
    
    # draw blue circle at centroid
    if not isnan(cx) and not isnan(cy):
        cv.circle(frame,(int(cx), int(cy)), 6, (255,0,0), -1)

    # if any circles were found
    if avg_circ_center is not None and frame_n > 1:
        if not init:
            # initialize predictions for Kalman Filter
            kf.statePre = np.array([[avg_circ_center[0]], [avg_circ_center[1]], [0], [0]], np.float32)
            kf.statePost = np.array([[avg_circ_center[0]], [avg_circ_center[1]], [0], [0]], np.float32)
            init = True
        # use measurement to correct prediction (this same method can be used for each available sensor)
        kf.correct(np.array(avg_circ_center, np.float32))

    # calculate and draw Kalman prediction as green circle with black outline            
    if init:
        pred_arr = kf.predict()
        cv.circle(frame,(int(pred_arr[0]),int(pred_arr[1])), 10, (0,0,0), -1)
        cv.circle(frame,(int(pred_arr[0]),int(pred_arr[1])), 6, (0,255,0), -1)

    # display windows
    cv.imshow('Original', frame)
    #cv.imshow('Color', tr)
    #cv.imshow('FG mask (color)', fg_color)
    #cv.imshow('FG (all)', frame_mov)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()