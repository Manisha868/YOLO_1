#CMD code
#trian
#yolo task=detect mode=train epochs=20 data=data.yaml model=yolov8n.pt imgsz=320 batch=4 workers=1 amp=False







#python code

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/manisha/Desktop/yolov8/repo1/yolodata/data.yaml", epochs=20, imgsz=320, batch=4, workers=1, amp=False)  # train the model

