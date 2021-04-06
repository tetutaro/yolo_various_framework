#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx


# export-friendly version of nn.SiLU()
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# export-friendly version of nn.Hardswish()
class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        # for torchscript and CoreML
        # return x * F.hardsigmoid(x)
        # for torchscript, CoreML and ONNX
        return x * F.hardtanh(x + 3, 0., 6.) / 6.


def yolov5_convert_torch_onnx(
    model: str,
    directory: str,
    imgsize: List[int],
    repo: str = 'ultralytics/yolov5:v4.0'
) -> None:
    path_weight = f'{directory}/{model}.pt'
    if not os.path.isfile(path_weight):
        return
    path_torch = f'{directory}/{model}.pth'
    path_onnx = f'{directory}/{model}.onnx'
    dummy_image = torch.zeros(1, 3, *imgsize)
    model_torch = torch.hub.load(repo, model)
    ckpt = torch.load(
        path_weight, map_location='cpu'
    )['model']
    model_torch.load_state_dict(ckpt.state_dict())
    model_torch.names = ckpt.names
    # save state dict
    if not os.path.isfile(path_torch):
        torch.save(model_torch.state_dict(), path_torch)
    if os.path.isfile(path_onnx):
        return
    model_torch_onnx = model_torch.fuse()
    model_torch_onnx.eval()
    for k, m in model_torch_onnx.named_modules():
        m._non_persistent_buffers_set = set()
        if m.__class__.__name__ == 'Conv':
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model_torch_onnx.model[-1].export = True
    _ = model_torch_onnx(dummy_image)
    print('Starting ONNX export with onnx %s...' % onnx.__version__)
    try:
        torch.onnx.export(
            model_torch_onnx, dummy_image, path_onnx,
            verbose=False,
            opset_version=12,
            input_names=['images'],
            output_names=['output', 'output_1', 'output_2']
        )
        model_onnx = onnx.load(path_onnx)
        onnx.checker.check_model(model_onnx)
        print('ONNX export success: %s' % path_onnx)
    except Exception as e:
        print('ONNX export failure: %s' % e)
    return
