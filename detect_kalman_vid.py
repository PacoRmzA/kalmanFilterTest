import cv2
import numpy as np
from math import isnan

STABILIZATION_ENABLED = True
HOUGH_ENABLED = True
KALMAN_ENABLED = True
ARM_ENABLED = True
MORPH_EX = True

# the video had to be stabilized and scaled to make foreground detection effective
cap = cv2.VideoCapture('video_cel_kalman_stabilized.mp4') \
    if STABILIZATION_ENABLED else cv2.VideoCapture('video_cel_kalman.mp4')
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
backSub = cv2.createBackgroundSubtractorMOG2(100, 300, False) # background subtractor, 100 frame history, 300 sensitivity, no shadows
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 3px circle kernel for morphological operations

if KALMAN_ENABLED:
    k = cv2.KalmanFilter(4,2) # 4 state vars (pos xy, vel xy), 2 measurement vars (pos xy)
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
                [[7, 0],
                [0, 7]], np.float32)

    init = False

if cap.isOpened() == False:
    print("Error opening file")

while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        break

    # arm and face color mask
    frame_arm = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (0, 0, 151), (180, 222, 255))

    # foreground/movement mask
    fg_mask = backSub.apply(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    # detects circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=20) if HOUGH_ENABLED else None

    circ_found = False

    if circles is not None: # if any circles were found
        circ_found = True
        n_circ = 0 # number of circles
        avg_circ_x = 0
        avg_circ_y = 0
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            n_circ += 1
            avg_circ_x += i[0]
            avg_circ_y += i[1]
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3) # draw magenta circle
        avg_circ_x /= n_circ # avg circle center x position
        avg_circ_y /= n_circ # avg circle center y position

    if MORPH_EX:
        if ARM_ENABLED:
            # increase size of arm mask
            cv2.dilate(frame_arm, kernel, frame_arm, iterations=2)

        # apply opening operation to foreground (remove noise and enlarge detection)
        cv2.erode(fg_mask, kernel, fg_mask, iterations=4)
        cv2.dilate(fg_mask, kernel, fg_mask, iterations=2)


    # movement minus arm should leave only phone
    frame_final = fg_mask - frame_arm if ARM_ENABLED else fg_mask

    if MORPH_EX:
        # opening is also applied for final frame, but with more erosion and less dilation
        cv2.erode(frame_final, kernel, frame_final, iterations=1)
        cv2.dilate(frame_final, kernel, frame_final, iterations=4)

    # centroid is calculated as the average of all white pixels in the final mask
    segmentation = np.where(frame_final == 255)
    cx = np.mean(segmentation[1]) 
    cy = np.mean(segmentation[0]) 

    if not isnan(cx) and not isnan(cy):
        # if circles are found these are considered much more important than the centroid
        if circ_found:
            cx = 0.25*cx + 0.75*avg_circ_x
            cy = 0.25*cy + 0.75*avg_circ_y
        if KALMAN_ENABLED:
            if not init:
                # initialize predictions for Kalman Filter
                k.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                k.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                init = True
            k.correct(np.array([cx, cy], np.float32)) # use measurement to correct prediction

    if KALMAN_ENABLED and init:
        pred_arr = k.predict() # use model to predict phone position and draw it as a yellow circle
        cv2.circle(frame,(int(pred_arr[0]),int(pred_arr[1])), 10, (0,0,0), -1)
        cv2.circle(frame,(int(pred_arr[0]),int(pred_arr[1])), 6, (0,255,255), -1)
    else:
        cv2.circle(frame,(int(cx), int(cy)), 10, (0,0,0), -1)
        cv2.circle(frame,(int(cx), int(cy)), 6, (0,255,255), -1)

    cv2.imshow('Original', frame)
    cv2.imshow('Foreground/Movement mask', fg_mask)
    if ARM_ENABLED:
        cv2.imshow('Arm color threshold', frame_arm)
        cv2.imshow('Fg/Mask minus arm color', frame_final)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
