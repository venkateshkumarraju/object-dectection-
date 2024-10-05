# Vehicle Registration Plate Detection
This project is an implementation of an object detection model that detects vehicle registration plates in images. The dataset used for this project consists of labeled images of vehicle registration plates. We employ YOLOv5 for training and evaluation, and the model is evaluated using COCO detection metrics.

## Table of Contents
Dataset

Installation

Training

Inference

Evaluation

Run Inference on a Video

Results

## Dataset
The dataset used in this project contains vehicle registration plates and their corresponding bounding box labels. The dataset is split into training and validation sets with the following structure:
```
Dataset/
├── train/
│   └── Vehicle registration plate/
│       └── Label/
└── validation/
    └── Vehicle registration plate/
        └── Label/
```
## Dataset Structure
Each label file contains bounding box information in the format:

Vehicle registration plate xmin ymin xmax ymax
Where xmin, ymin, xmax, and ymax represent the coordinates of the bounding box.  and it is convert to yolo format

For each image, create a corresponding text file (usually with the .txt extension).
In this text file, define bounding boxes for the objects present in the image.
Each line in the file represents an object and contains the following information:
Class index (usually zero-indexed, starting from 0).
Bounding box coordinates (normalized xywh format, ranging from 0 to 1).


# Example
Class index x_center y_center width height  

0 0.4902 0.4592 0.7516 0.3693  



## Installation
To run this project, clone the YOLOv5 repository and install the required dependencies:

```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```
## Training


write custom dataset  data. yaml file .The data.yaml file should point to the dataset location:
```
%%writefile data.yaml
path: '/path/to/dataset'
train: '/path/to/dataset/train/Vehicle registration plate'
val: '/path/to/dataset/validation/Vehicle registration plate'
nc: 1
names: ['reg plate']
```
To train the YOLOv5 model on the vehicle registration plate dataset, use the following command:
```
python train.py --data data.yaml --weights yolov5m.pt --img 640 --epochs 30 --batch-size 64
``` 



## Inference
Once the model is trained, you can perform inference on validation images by running:



bash
```
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.5 --source '/path/to/validation/images'
```
This will save the predicted bounding boxes in both image and text formats.

![image](https://github.com/user-attachments/assets/15ebcc6d-256d-49f4-8e78-07815d2b5b63)

## Evaluation
The trained model is evaluated using the COCO detection metrics. To evaluate the model, use the following command:

```
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt --img 640 --batch-size 32 --save-json
```
## Run Inference on a Video
You can also run inference on a video file. Download the input video from here and use the following command:

```
python detect.py --weights runs/train/exp/weights/best.pt --source '/path/to/video'
```
## Video Output:
YouTube link to output video
https://youtu.be/8MEb1RHeghA


## COCO Evaluation Results:

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.650
 
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.930
 
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.748
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.717
 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.751
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.577
...
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.467
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.784
 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.803
 
COCO Metrics: [0.650304695594039, 0.9302446889486897, 0.7484658273933946, 0.3381050586050572, 0.7166377599305075, 0.7514961561810386, 0.57734375, 0.7125, 0.7150390625, 0.46747967479674796, 0.7843137254901961, 0.8032432432432433]
