#!/usr/bin/env python
# -*- coding:utf-8 -*-
from convertor.yolov5.convert_onnx import yolov5_convert_onnx
from convertor.yolov5.convert_onnx_vino import yolov5_convert_onnx_vino
from convertor.yolov5.convert_onnx_tf import yolov5_convert_onnx_tf
from convertor.yolov5.convert_tf_tflite import yolov5_convert_tf_tflite
from convertor.yolov5.convert_tf_onnx import yolov5_convert_tf_onnx

IMAGE_SIZE = 640
DIRECTORY = 'weights/yolov5'


if __name__ == '__main__':
    for x in ['s', 'm', 'l', 'x']:
        model = f'yolov5{x}'
        yolov5_convert_onnx(
            model=model,
            directory=DIRECTORY,
            imgsize=[IMAGE_SIZE, IMAGE_SIZE]
        )
        yolov5_convert_onnx_vino(
            model=model,
            directory=DIRECTORY,
            imgsize=[IMAGE_SIZE, IMAGE_SIZE]
        )
        yolov5_convert_onnx_tf(
            model=model,
            directory=DIRECTORY
        )
        yolov5_convert_tf_tflite(
            model=model,
            directory=DIRECTORY,
            imgsize=[IMAGE_SIZE, IMAGE_SIZE]
        )
        yolov5_convert_tf_onnx(
            model=model,
            directory=DIRECTORY
        )
