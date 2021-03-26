#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Dict
import os
import yaml
# Torch
import torch
# TensorFlow
import tensorflow as tf
# YOLO V5
from models.tf_yolov5 import tf_YoloV5, tf_Detect
from utils.convert_tflite import (
    save_frozen_graph,
    convert_tflite_fp32,
    convert_tflite_fp16,
    convert_tflite_int8,
)


def _convert_tf_keras_model(
    model: str,
    imgsize: List[int],
    model_torch: torch.nn.Module,
    nclasses: int,
    config: Dict
) -> tf.keras.Model:
    model_tf = tf_YoloV5(
        model_torch=model_torch,
        nclasses=nclasses,
        config=config
    )
    m = model_tf.model.layers[-1]
    assert isinstance(m, tf_Detect), "the last layer must be Detect"
    m.training = False
    # dummy run and check output
    dummy_image_tf = tf.zeros((1, *imgsize, 3))  # NHWC
    y = model_tf.predict(dummy_image_tf)
    for yy in y:
        _ = yy.numpy()
    # create keras model
    inputs_keras = tf.keras.Input(
        shape=(*imgsize, 3), batch_size=1
    )
    outputs_keras = model_tf.predict(inputs=inputs_keras)
    model_keras = tf.keras.Model(
        inputs=inputs_keras,
        outputs=outputs_keras,
        name=model
    )
    # model_keras.summary()
    return model_keras


def yolov5_convert_tf_tflite(
    model: str,
    directory: str,
    imgsize: List[int]
) -> None:
    path_weights = f'{directory}/{model}.pt'
    if not os.path.isfile(path_weights):
        print(f'ERROR: {path_weights} not found')
        return
    # dummy image
    dummy_image_torch = torch.zeros((1, 3, *imgsize))  # NCHW
    # Load PyTorch model
    model_torch = torch.load(
        path_weights,
        map_location='cpu'
    )['model'].float()  # .fuse()
    model_torch.eval()
    # export=True to export Detect Layer
    model_torch.model[-1].export = False
    # dry run
    y = model_torch(dummy_image_torch)
    # number of classes
    nclasses = y[0].shape[-1] - 5
    # load configuration for the model
    path_config = f'models/{model}.yaml'
    with open(path_config, 'rt') as rf:
        config = yaml.safe_load(rf)
    # TensorFlow Keras export
    model_keras = _convert_tf_keras_model(
        model=model,
        imgsize=imgsize,
        model_torch=model_torch,
        nclasses=nclasses,
        config=config
    )
    # save as Frozen Graph
    save_frozen_graph(
        path_pb=f'{directory}/{model}.pb', model_keras=model_keras
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
