#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List
import os
import subprocess

OPENVINO_VERSION = "2021"


def yolov5_convert_onnx_vino(
    model: str,
    directory: str,
    imgsize: List[int]
) -> None:
    path_onnx = f'{directory}/{model}.onnx'
    dir_vino = f'{directory}/onnx_vino_{model}'
    if not os.path.isfile(path_onnx):
        return
    if os.path.isdir(dir_vino):
        return
    path_mo = f'/opt/intel/openvino_{OPENVINO_VERSION}/'
    path_mo += 'deployment_tools/model_optimizer/mo.py'
    options = list()
    options.append(f'--input_model {path_onnx}')
    options.append(f'--model_name {model}')
    options.append(f'--output_dir {dir_vino}')
    options.append('--data_type FP32')
    options.append('--input images')
    options.append(f'--input_shape [1,3,{imgsize[0]},{imgsize[1]}]')
    cmd = ' '.join(['python', path_mo] + options)
    cmd = cmd.split()
    subprocess.run(cmd)
    return
