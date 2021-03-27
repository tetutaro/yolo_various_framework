#!/usr/bin/env python
# -*- cofing:utf-8 -*-
import os
import subprocess


def yolo_convert_tf_onnx(
    model: str,
    directory: str
) -> None:
    path_pb = f'{directory}/{model}.pb'
    path_onnx = f'{directory}/tf_{model}.onnx'
    if not os.path.isfile(path_pb):
        return
    if os.path.isfile(path_onnx):
        return
    options = list()
    options.append(f'--graphdef {path_pb}')
    options.append(f'--output {path_onnx}')
    options.append('--inputs x:0')
    if model in ['yolov3-tiny', 'yolov4-tiny']:
        options.append('--outputs Identity:0,Identity_1:0')
    else:
        options.append('--outputs Identity:0,Identity_1:0,Identity_2:0')
    options.append('--opset 12')
    options.append('--inputs-as-nchw x:0')
    cmd = ' '.join(['python -m tf2onnx.convert'] + options)
    cmd = cmd.split()
    subprocess.run(cmd)
    return
