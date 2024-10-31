import cv2 as cv
import numpy as np

net = cv.dnn.readNetFromONNX("/home/thecubicjedi/train_yolo_screwdriver/train2/weights/best.onnx")

INPUT_WIDTH = 640
INPUT_HEIGHT = 640


cap = cv.VideoCapture('/home/thecubicjedi/train_yolo_screwdriver/test_vid.mp4')
fps = cap.get(cv.CAP_PROP_FPS)
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter('output.mp4', fourcc, fps, (width, height))

while True:
    
    ret, frame = cap.read()
    if frame is None:
        break

    blob = cv.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=True)
    net.setInput(blob)

    output = net.forward()
    preds = output.transpose((0, 2, 1))

    # Extract output detection
    class_ids, confs, boxes = list(), list(), list()

    image_height, image_width, _ = frame.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    rows = preds[0].shape[0]

    for i in range(rows):
        row = preds[0][i]
        conf = row[4]
        
        classes_score = row[4:]
        _,_,_, max_idx = cv.minMaxLoc(classes_score)
        class_id = max_idx[1]
        if (classes_score[class_id] > .25):
            confs.append(conf)
            label = int(class_id)
            class_ids.append(label)
            
            #extract boxes
            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])
            boxes.append(box)
            
    r_class_ids, r_confs, r_boxes = list(), list(), list()

    indexes = cv.dnn.NMSBoxes(boxes, confs, 0.25, 0.45) 
    for i in indexes:
        r_class_ids.append(class_ids[i])
        r_confs.append(confs[i])
        r_boxes.append(boxes[i])

    for i in indexes:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        conf = confs[i]
        
        cv.rectangle(frame, (left, top), (left + width, top + height), (0,255,0), 3)

    cv.imshow("frame", frame)
    out.write(frame)
    
    key = cv.waitKey(10)
    if key == ord('q') or key == 27:
        break

cap.release()
out.release()
cv.destroyAllWindows()

