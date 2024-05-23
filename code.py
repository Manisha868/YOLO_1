#training
yolo task=detect mode=train epochs=20 data=data.yaml model=yolov8n.pt imgsz=320 batch=4 workers=1 amp=False

#test
#yolo 
