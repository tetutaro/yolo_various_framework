#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import onnx
from onnx_tf.backend import prepare


def yolov5_convert_onnx_tf(model: str, directory: str) -> None:
    path_onnx = f'{directory}/{model}.onnx'
    if not os.path.isfile(path_onnx):
        return
    path_onnx_tf = f'{directory}/onnx_tf_{model}'
    if os.path.isdir(path_onnx_tf):
        return
    model_onnx = onnx.load(path_onnx)
    tf_rep = prepare(model_onnx)
    tf_rep.export_graph(path_onnx_tf)
    return
