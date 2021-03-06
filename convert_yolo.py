#!/usr/bin/env python
# -*- coding:utf-8 -*-
# from convertor.yolo.convert_torch_onnx import yolo_convert_torch_onnx
# from convertor.yolo.convert_onnx_vino import yolo_convert_onnx_vino
# from convertor.yolo.convert_onnx_tf import yolo_convert_onnx_tf
from convertor.yolo.convert_tf_tflite import yolo_convert_tf_tflite
from convertor.yolo.convert_tf_onnx import yolo_convert_tf_onnx

IMAGE_SIZES = {
    'yolov3-tiny': 512,
    'yolov3': 512,
    'yolov3-spp': 512,
    'yolov4-tiny': 512,
    'yolov4': 512,
    'yolov4-csp': 512,
    'yolov4x-mish': 512,
}
DIRECTORY = 'weights/yolo'


if __name__ == '__main__':
    for model in [
        'yolov3-tiny', 'yolov3', 'yolov3-spp',
        'yolov4-tiny', 'yolov4',
        # 'yolov4-csp', 'yolov4x-mish',
    ]:
        imgsize = IMAGE_SIZES[model]
        # yolo_convert_torch_onnx(
        #     model=model,
        #     directory=DIRECTORY,
        #     imgsize=[imgsize, imgsize]
        # )
        # yolo_convert_onnx_vino(
        #     model=model,
        #     directory=DIRECTORY,
        #     imgsize=[imgsize, imgsize]
        # )
        # yolo_convert_onnx_tf(
        #     model=model,
        #     directory=DIRECTORY
        # )
        yolo_convert_tf_tflite(
            model=model,
            directory=DIRECTORY,
            imgsize=[imgsize, imgsize]
        )
        yolo_convert_tf_onnx(
            model=model,
            directory=DIRECTORY
        )
