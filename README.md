
     Object detection Using Yolov8 Model on Custom Dataset






## [YOLOV8](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)

YOLOv8 is the latest version of the YOLO (You Only Look Once) AI models developed by Ultralytics. This notebook serves as the starting point for exploring the various resources available to help you get started with YOLOv8 and understand its features and capabilities.

YOLOv8 models are fast, accurate, and easy to use, making them ideal for various object detection and image segmentation tasks. They can be trained on large datasets and run on diverse hardware platforms, from CPUs to GPUs.


##  [Dataset](https://github.com/Manisha868/YOLO_1/tree/main/yolodata)

The dataset has been created by me. First collect the images form surrounding. Thereafter, they were annotated carefully using free labelling [software](https://universe.roboflow.com/) available online. 
After that dataset was downloaded in zip foramt and then extracted.


This dataset has 2 folders:-
 
1. [images](https://github.com/Manisha868/YOLO_1/tree/main/yolodata/images) :- Contains 3 folders [train](https://github.com/Manisha868/YOLO_1/tree/main/yolodata/images/train), [test](https://github.com/Manisha868/YOLO_1/tree/main/yolodata/images/test), [val](https://github.com/Manisha868/YOLO_1/tree/main/yolodata/images/val).


![0](https://github.com/Manisha868/YOLO_1/blob/main/yolodata/images/train/WhatsApp-Image-2024-05-22-at-14-16-04-1-_jpeg.rf.15150e49fc2cd77623d81a52e5d19323.jpg?raw=true)
Contains total 53 images form which 37 images for training, 5 images for test and 11 images selected for validation.

2. [labels](https://github.com/Manisha868/YOLO_1/tree/main/yolodata/labels):- There are 17 classes on the dataset ( 0: candle
  1: cip
  2: comb
  3: earser
  4: fock
  5: fstick
  6: hairclip
  7: key
  8: knife
  9: lock
  10: marker
  11: mirror
  12: mouse
  13: pen
  14: sharpner
  15: spoon
  16: statue)


## Preparing the configuration [YAML file](https://github.com/Manisha868/YOLO_1/blob/main/yolodata/data.yaml)


In order to train a YOLOv8 model for object detection, we need to provide specific configurations such as the dataset path, classes and training and validation sets. These configurations are typically stored in a YAML (Yet Another Markup Language) file which serves as a single source of truth for the model training process. This allows for easy modification and replication of the training process, as well as providing a convenient way to store and manage configuration settings.

This YAML file should follow this format:

```bash
  path: /yolov8/dataset/data  # Use absolute path 
  train: images/train
  test: images/test
  val: images/val

# Classes
  names:
   0: candle
   1: cip
   2: comb
   3: earser
   4: fock

```




## [Train the Model](https://github.com/Manisha868/YOLO_1/blob/main/yolodata/train_yolo.py)

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# train the model
model.train(data="/home/manisha/Desktop/yolov8/repo1/yolodata/data.yaml", epochs=20, imgsz=320, batch=4, workers=1, amp=False)  # train the model

```

## [Inspecting training results](https://github.com/Manisha868/YOLO_1/tree/main/yolodata/runs/detect)


The train method automatically saves the results in ./runs/detect/train. 
![1](https://github.com/Manisha868/YOLO_1/blob/main/yolodata/runs/detect/train/train_batch101.jpg?raw=true)

                          with labels
![2](https://github.com/Manisha868/YOLO_1/blob/main/yolodata/runs/detect/train/val_batch0_labels.jpg?raw=true)


