#!/usr/bin/env bash

if [ ! -d yolo ] ; then
    mkdir yolo
fi
if [ ! -f yolo/yolov3-tiny.weights ] ; then
    wget -O yolo/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights
fi
if [ ! -f yolo/yolov3.weights ] ; then
    wget -O yolo/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
fi
if [ ! -f yolo/yolov4-tiny.weights ] ; then
    wget -O yolo/yolov4-tiny.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
fi
if [ ! -f yolo/yolov4.weights ] ; then
    wget -O yolo/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
fi
if [ ! -f yolo/yolov4-csp.weights ] ; then
    wget -O yolo/yolov4-csp.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights
fi
if [ ! -f yolo/yolov4x-mish.weights ] ; then
    wget -O yolo/yolov4x-mish.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights
fi
