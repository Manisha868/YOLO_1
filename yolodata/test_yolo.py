#CMD code
#test
#yolo task=detect mode=predict model=yolov8n.pt show=True conf=0.5 source=1.jpg


#python code

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  
model.predict(source="1.jpg", show=True, save=True, conf=0.5) #test the model

