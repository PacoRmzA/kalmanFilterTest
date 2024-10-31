from ultralytics import YOLO

model = YOLO("/home/thecubicjedi/train_yolo_screwdriver/train2/weights/best.pt")

model.export(format="onnx", imgsz=[640,640], opset=12, simplify=True)