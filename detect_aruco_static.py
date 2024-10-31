import cv2 as cv

frame = cv.imread("arucos/aruco_0.png")

# Convert the frame to grayscale
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
parameters = cv.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
# Detect the markers
corners, ids, rejected = detector.detectMarkers(gray)
# Print the detected markers
#print("Detected markers:", ids)
if ids is not None:
    cv.aruco.drawDetectedMarkers(frame, corners, ids)

cv.imshow("frame", frame)

key = cv.waitKey(0)