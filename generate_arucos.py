import cv2 as cv
import numpy as np

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

for i in range(250):
    img = cv.aruco.generateImageMarker(aruco_dict, i, 200)
    img = np.pad(img, pad_width=100, constant_values=255)
    cv.imwrite("arucos/aruco_"+str(i)+".png", img)