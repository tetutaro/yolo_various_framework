#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List
import tensorflow as tf
import os
import math
import numpy as np
# YOLO V3, V4
from models.tf_yolo import (
    tf_YoloV3_tiny,
    tf_YoloV3,
    tf_YoloV3_spp,
    tf_YoloV4_tiny,
    tf_YoloV4,
    tf_YoloV4_csp,
    tf_YoloV4x_mish,
)
from utils.convert_tflite import (
    save_frozen_graph,
    convert_tflite_fp32,
    convert_tflite_fp16,
    convert_tflite_int8,
)

NUM_CLASS = 80
MODEL_CLASS = {
    'yolov3-tiny': tf_YoloV3_tiny,
    'yolov3': tf_YoloV3,
    'yolov3-spp': tf_YoloV3_spp,
    'yolov4-tiny': tf_YoloV4_tiny,
    'yolov4': tf_YoloV4,
    'yolov4-csp': tf_YoloV4_csp,
    'yolov4x-mish': tf_YoloV4x_mish,
}
MODEL_SHAPE = {
    'yolov3-tiny': {
        'nlayers': 13,
        'nobn_layers': [9, 12],
    },
    'yolov3': {
        'nlayers': 75,
        'nobn_layers': [58, 66, 74],
    },
    'yolov3-spp': {
        'nlayers': 76,
        'nobn_layers': [59, 67, 75],
    },
    'yolov4-tiny': {
        'nlayers': 21,
        'nobn_layers': [17, 20],
    },
    'yolov4': {
        'nlayers': 110,
        'nobn_layers': [93, 101, 109],
    },
    'yolov4-csp': {
        'nlayers': 115,
        'nobn_layers': [94, 104, 114],
    },
    'yolov4x-mish': {
        'nlayers': 137,
        'nobn_layers': [112, 124, 136],
    },
}
STRIDES = [8, 16, 32, 64, 128]
DEBUG = False


def _load_darknet_weights(
    model: str,
    path_weights: str,
    model_keras: tf.keras.Model
) -> None:
    rf = open(path_weights, 'rb')
    major, minor, revision, seen, _ = np.fromfile(
        rf, dtype=np.int32, count=5
    )
    nlayers = MODEL_SHAPE[model]['nlayers']
    nobn_layers = MODEL_SHAPE[model]['nobn_layers']
    if DEBUG:
        print(model)
        for i, layers in enumerate(model_keras.weighted_layers):
            print(i, layers)
    assert len(model_keras.weighted_layers) == nlayers
    for i, layers in enumerate(model_keras.weighted_layers):
        conv_layer = layers.conv
        norm_layer = layers.norm
        input_shape = layers.input_shape
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = input_shape[-1]
        if i not in nobn_layers:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(
                rf, dtype=np.float32, count=(4 * filters)
            )
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
        else:
            conv_bias = np.fromfile(
                rf, dtype=np.float32, count=filters
            )
            if model in ['yolov4-csp', 'yolov4x-mish']:
                # https://arxiv.org/abs/1708.02002
                offset = np.where(np.array(nobn_layers) == i)[0][0]
                stride = STRIDES[offset]
                conv_bias = conv_bias.reshape(3, 85)
                conv_bias[:, 4] += math.log(8 / (640 / stride) ** 2)
                conv_bias[:, 5:] += math.log(0.6 / (80 - 0.99))
                conv_bias = conv_bias.reshape(-1)
        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(
            rf, dtype=np.float32, count=np.product(conv_shape)
        )
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose(
            [2, 3, 1, 0]
        )
        if i not in nobn_layers:
            assert norm_layer.__class__.__name__ == 'BatchNormalization'
            conv_layer.set_weights([conv_weights])
            norm_layer.set_weights(bn_weights)
        else:
            assert norm_layer.__class__.__name__ == 'function'
            conv_layer.set_weights([conv_weights, conv_bias])
    rest = len(rf.read())
    assert rest == 0, f'failed to read all data: {rest}'
    rf.close()
    return


def yolo_convert_tf_tflite(
    model: str,
    directory: str,
    imgsize: List[int]
) -> None:
    path_weights = f'{directory}/{model}.weights'
    if not os.path.isfile(path_weights):
        print(f'ERROR: {path_weights} not found')
        return
    # load model
    model_keras = MODEL_CLASS[model](nc=NUM_CLASS)
    model_keras.build(input_shape=(1, *imgsize, 3))
    # dummy run
    dummy_image_tf = tf.zeros((1, *imgsize, 3), dtype=tf.float32)  # NHWC
    y = model_keras(dummy_image_tf)
    for yy in y:
        _ = yy.numpy()
    # model_keras.summary()
    # load weights
    _load_darknet_weights(
        model=model, path_weights=path_weights, model_keras=model_keras
    )
    # save as Frozen Graph
    input_keras = tf.keras.Input(
        shape=(*imgsize, 3), batch_size=1, dtype=tf.float32
    )
    save_frozen_graph(
        path_pb=f'{directory}/{model}.pb',
        model_keras=model_keras,
        input_keras=input_keras
    )
    # convert TFLite model
    path_tflite = f'{directory}/{model}_fp32.tflite'
    convert_tflite_fp32(path_tflite=path_tflite, model_keras=model_keras)
    path_tflite = f'{directory}/{model}_fp16.tflite'
    convert_tflite_fp16(path_tflite=path_tflite, model_keras=model_keras)
    path_tflite = f'{directory}/{model}_int8.tflite'
    convert_tflite_int8(
        path_tflite=path_tflite,
        imgsize=imgsize,
        model_keras=model_keras
    )
    return
