#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import argparse
from detector.base import Config
from detector.yolov5 import DetectorYoloV5
from detector.yolo import DetectorYolo


def main(config: Config) -> None:
    if config.model.startswith(('yolov3', 'yolov4')):
        detector = DetectorYolo(config=config)
    elif config.model.startswith('yolov5'):
        detector = DetectorYoloV5(config=config)
    else:
        raise SystemError(f'model is incorrect ({config.model})')
    detector.print_header()
    for sess in detector.yield_session():
        detector.inference(sess=sess)
        detector.print_result(sess=sess)
        detector.dump_result(sess=sess)
        detector.dump_image(sess=sess)
    detector.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='detect objects from images'
    )
    parser.add_argument(
        '-m', '--model', type=str, required=True, choices=[
            'yolov3-tiny', 'yolov3',
            'yolov4-tiny', 'yolov4', 'yolov4-csp', 'yolov4x-mish',
            'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x',
        ], help='model name'
    )
    parser.add_argument(
        '-f', '--framework', type=str, required=True, choices=[
            'torch', 'torch_onnx', 'onnx_vino', 'onnx_tf',
            'tf', 'tflite', 'tf_onnx'
        ], help='framework'
    )
    parser.add_argument(
        '-q', '--quantize', type=str, default='fp32', choices=[
            'fp32', 'fp16', 'int8'
        ], help='quantization mode (TensorFlow Lite only)'
    )
    parser.add_argument(
        '-d', '--image-dir', type=str, required=True,
        help='directory contains images to detect objects'
    )
    parser.add_argument(
        '-c', '--conf-threshold', type=float, default=0.3,
        help='threshold of confidence score to adopt bounding boxes'
    )
    parser.add_argument(
        '-i', '--iou-threshold', type=float, default=0.45,
        help='threshold of IoU to eliminte bounding boxes in NMS'
    )
    parser.add_argument(
        '--disable-clarify-image', action='store_true',
        help='disable image preprocessing'
    )
    parser.add_argument(
        '--disable-use-superres', action='store_true',
        help='disable using Super-Resolution at image preprocessing'
    )
    parser.add_argument(
        '--disable-soft-nms', action='store_true',
        help='use hard-NMS instead of soft-NMS'
    )
    parser.add_argument(
        '--disable-iou-subset', action='store_true',
        help=(
            'do not eliminate small and unconfident bounding box'
            ' which is inside of big and confident bounding box'
        )
    )
    args = parser.parse_args()
    if not os.path.isdir(args.image_dir):
        raise ValueError(
            f'image directory not found ({args.image_dir})'
        )
    if (args.conf_threshold < 0.0) or (args.conf_threshold >= 1.0):
        raise ValueError(
            f'confidence threshold is incorrect ({args.conf_threshold})'
        )
    if (args.iou_threshold < 0.0) or (args.iou_threshold >= 1.0):
        raise ValueError(
            f'IoU threshold is incorrect ({args.iou_threshold})'
        )
    config = Config(**vars(args))
    main(config=config)
