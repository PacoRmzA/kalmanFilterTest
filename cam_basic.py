import cv2 as cv

cap = cv.VideoCapture(1)

while True:
    
    ret, frame = cap.read()
    if frame is None:
        break

    cv.imshow("frame", frame)
    
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break

