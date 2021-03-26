# yolo_various_platforms

run YOLO (object detection model) on various platforms and compare them

## my motivations

- I want to run object detection models on my PC
    - I don't have so much money to buy any GPU
- I want to know which platform is the best in the meaning of elapsed time
- I want to confirm that the predicted results are not so much different when I convert pre-trained weights to another platform

## object deteciton models

- [YOLO V3](https://github.com/pjreddie/darknet)
    - yolov3-tiny
        - input image size: 512x512
    - yolov3
        - input image size: 512x512
- [YOLO V4](https://github.com/AlexeyAB/darknet)
    - yolov4-tiny
        - input image size: 512x512
    - yolov4
        - input image size: 512x512
    - yolov4-csp
        - input image size: 640x640
    - yolov4x-mish
        - input image size: 640x640
- [YOLO V5](https://github.com/ultralytics/yolov5)
    - yolov5s
        - input image size: 640x640
    - yolov5m
        - input image size: 640x640
    - yolov5l
        - input image size: 640x640
    - yolov5x
        - input image size: 640x640

## deep learning platforms

all deep learing platforms below were ran on Python 3.7.9

- PyTorch
    - torch 1.7.1
- TensorFlow (Frozen Graph)
    - tensorflow 2.4.1
- TensorFlow Lite
    - tflite-runtime 2.5.0
- ONNX
    - onnxruntime 1.6.0
- OpenVINO
    - OpenVINO 2021.2.185

## libraries to convert

- onnx 1.8.1
- onnx-tf 1.7.0
- tf2onnx 1.8.3

## my environment

- MacBook Air (Retina, 2020)
    - CPU: 1.1GHz quad core Intel Core i5
    - Memory: 16GB 3733MHz LPDDR4X

## preparation

- download font for detector
    - `./download_font.sh`
- download COCO dataset and create small dataset (convert annotations)
    - see datasets/README
- download pre-trained weights
    - see weights/README
- convert pre-trained weights to various platforms
    - `./convert_yolo.py`
    - `./convert_yolov5.py`
- (if you want to calc metrics) please install [`object_detection_metrics`](https://github.com/tetutaro/object_detection_metrics)
    - `> pip install "git+https://github.com/tetutaro/object_detection_metrics.git`

## usage

```
detect.py
```
