#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union
import math
# Torch
import torch.nn as nn
# TensorFlow
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, swish
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import (
    Layer, BatchNormalization, Conv2D, MaxPool2D
)
# YOLO V5
from models.common import (
    autopad,
    Conv, Bottleneck, BottleneckCSP, C3,
    SPP, Focus, Concat,
)
from models.yolo import Detect


def make_divisible(x: int, divisor: int) -> int:
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


# wrapper of TensorFlow BatchNormalization
# (convert torch.nn._BatchNorm -> tf.keras.layers.BatchNormalization)
class tf_BN(Layer):
    def __init__(
        self: tf_BN,
        module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        assert module is not None
        self.bn = BatchNormalization(
            beta_initializer=Constant(
                module.bias.numpy()
            ),
            gamma_initializer=Constant(
                module.weight.numpy()
            ),
            moving_mean_initializer=Constant(
                module.running_mean.numpy()
            ),
            moving_variance_initializer=Constant(
                module.running_var.numpy()
            )
        )
        return

    def call(self: tf_BN, inputs: tf.Tensor) -> tf.Tensor:
        return self.bn(inputs)


class tf_Pad(Layer):
    def __init__(self: tf_Pad, pad: int) -> None:
        super().__init__()
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        return

    def call(self: tf_Pad, inputs: tf.Tensor) -> tf.Tensor:
        return tf.pad(
            inputs, self.pad, mode='constant', constant_values=0
        )


# wrapper of YOLO V5 Convolution Layer (models.common.Conv)
class tf_Conv(Layer):
    def __init__(
        self: tf_Conv,
        in_channels: int,
        out_channels: int,  # filters
        kernel_size: int = 1,
        strides: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activate: bool = True,
        module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        assert module is not None
        # TF v2.2 Conv2D does not support 'groups' argument"
        assert groups == 1
        # Convolute multi kernels are not allowed
        assert isinstance(kernel_size, int)
        # Convolution
        # TensorFlow convolution padding is inconsistent
        # with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843
        # /comparing-conv2d-with-padding-between-tensorflow-and-pytorch
        if strides == 1:
            padding_tf = 'SAME'
        else:
            padding_tf = 'VALID'
        kernel_initializer = Constant(
            module.conv.weight.permute(2, 3, 1, 0).numpy()
        )
        conv = Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding_tf,
            use_bias=False,
            kernel_initializer=kernel_initializer
        )
        if strides == 1:
            self.conv = conv
        else:
            self.conv = Sequential([
                tf_Pad(autopad(kernel_size, padding)),
                conv
            ])
        # batch normalization
        if hasattr(module, 'bn'):
            self.bn = tf_BN(module=module.bn)
        else:
            self.bn = tf.indentity
        # activation
        if not activate:
            self.act = tf.identity
        elif isinstance(module.act, nn.LeakyReLU):
            self.act = (lambda x: relu(x, alpha=0.1))
        elif isinstance(module.act, nn.Hardswish):
            self.act = (lambda x: x * tf.nn.relu6(x + 3) * 0.166666667)
        elif isinstance(module.act, nn.SiLU):
            # self.act = (lambda x: x * keras.activations.sigmoid(x))
            self.act = (lambda x: swish(x))
            return

    def call(self: tf_Conv, inputs: tf.Tensor) -> tf.Tensor:
        return self.act(self.bn(self.conv(inputs)))


# wrapper of YOLO V5 Focus Layer (models.common.Focus)
class tf_Focus(Layer):
    def __init__(
        self: tf_Focus,
        in_channels: int,
        out_channels: int,  # filters
        kernel_size: int = 1,
        strides: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activate: bool = True,
        module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        assert module is not None
        self.conv = tf_Conv(
            module=module.conv,
            in_channels=in_channels * 4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            activate=activate
        )
        return

    def call(self: tf_Focus, inputs: tf.Tensor) -> tf.Tensor:
        # x(b,w,h,c) -> y(b,w/2,h/2,4c)
        return self.conv(tf.concat([
                inputs[:, ::2, ::2, :],
                inputs[:, 1::2, ::2, :],
                inputs[:, ::2, 1::2, :],
                inputs[:, 1::2, 1::2, :]
        ], 3))


# wrapper of YOLO V5 Bottleneck Layer (models.common.Bottleneck)
class tf_Bottleneck(Layer):
    def __init__(
        self: tf_Bottleneck,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        assert module is not None
        hidden_channels = int(out_channels * expansion)
        self.cv1 = tf_Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            strides=1,
            module=module.cv1
        )
        self.cv2 = tf_Conv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            groups=groups,
            module=module.cv2
        )
        self.add = shortcut and in_channels == out_channels
        return

    def call(self: tf_Bottleneck, inputs: tf.Tensor) -> tf.Tensor:
        x = self.cv2(self.cv1(inputs))
        if self.add:
            x += inputs
        return x


# wrapper of TensorFlow Conv2D
# (convert torch.nn.Conv2D -> tf.keras.layers.Conv2D)
class tf_Conv2d(Layer):
    def __init__(
        self: tf_Conv2d,
        in_channels: int,
        out_channels: int,  # filters
        kernel_size: int,
        strides: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        assert module is not None
        # TF v2.2 Conv2D does not support 'groups' argument
        assert groups == 1
        kernel_initializer = Constant(
            module.weight.permute(2, 3, 1, 0).numpy()
        )
        if use_bias:
            bias_initializer = Constant(
                module.bias.numpy()
            )
        else:
            bias_initializer = None
        self.conv = Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding='VALID',
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        return

    def call(self: tf_Conv2d, inputs: tf.Tensor) -> tf.Tensor:
        return self.conv(inputs)


# wrapper of YOLO V5 BottleneckCSP Layer (models.common.BottleneckCSP)
class tf_BottleneckCSP(Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self: tf_BottleneckCSP,
        in_channels: int,
        out_channels: int,
        bottolenecks: int = 1,
        shotcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        assert module is not None
        hidden_channels = int(out_channels * expansion)
        self.cv1 = tf_Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            strides=1,
            module=module.cv1,
        )
        self.cv2 = tf_Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            strides=1,
            use_bias=False,
            module=module.cv2
        )
        self.cv3 = tf_Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            use_bias=False,
            module=module.cv3
        )
        self.cv4 = tf_Conv(
            in_channels=hidden_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            module=module.cv4
        )
        self.bn = tf_BN(module=module.bn)
        self.act = lambda x: relu(x, alpha=0.1)
        self.m = Sequential([
            tf_Bottleneck(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                shortcut=shotcut,
                groups=groups,
                expansion=1.0,
                module=module.m[i]
            ) for i in range(bottolenecks)
        ])
        return

    def call(self: tf_BottleneckCSP, inputs: tf.Tensor):
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


# wrapper of YOLO V5 C3 Layer (models.common.C3)
# (CSP Bottleneck with 3 convolutions)
class tf_C3(Layer):
    def __init__(
        self: tf_C3,
        in_channels: int,
        out_channels: int,
        bottolenecks: int = 1,
        shotcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        assert module is not None
        hidden_channels = int(out_channels * expansion)
        self.cv1 = tf_Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            strides=1,
            module=module.cv1
        )
        self.cv2 = tf_Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            strides=1,
            module=module.cv2
        )
        self.cv3 = tf_Conv(
            in_channels=hidden_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            module=module.cv3
        )
        self.m = Sequential([
            tf_Bottleneck(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                shortcut=shotcut,
                groups=groups,
                expansion=1.0,
                module=module.m[i]
            ) for i in range(bottolenecks)
        ])
        return

    def call(self: tf_C3, inputs: tf.Tensor):
        return self.cv3(tf.concat(
            (self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3
        ))


# wrapper of YOLO V3 SPP Layer (models.common.SPP)
# (Spatial Pyramid Pooling)
class tf_SPP(Layer):
    def __init__(
        self: tf_SPP,
        in_channels: int,
        out_channels: int,
        pool_sizes: Tuple[int] = (5, 9, 13),
        module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        assert module is not None
        hidden_channels = in_channels // 2
        self.cv1 = tf_Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            strides=1,
            module=module.cv1
        )
        self.cv2 = tf_Conv(
            in_channels=hidden_channels * (len(pool_sizes) + 1),
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            module=module.cv2
        )
        self.m = [
            MaxPool2D(pool_size=x, strides=1, padding='SAME')
            for x in pool_sizes
        ]
        return

    def call(self: tf_SPP, inputs: tf.Tensor) -> tf.Tensor:
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


# wrapper of YOLO V5 Detect Layer (models.yolo.Detect)
class tf_Detect(Layer):
    def __init__(
        self: tf_Detect,
        nc: int = 80,
        anchors: List[int] = [],
        channels: List[int] = [],
        imgsize: List[int] = [640, 640],
        module: Optional[nn.Module] = None
    ):
        super().__init__()
        assert module is not None
        self.imgsize = imgsize
        self.stride = tf.convert_to_tensor(
            module.stride.numpy(), dtype=tf.float32
        )
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.convert_to_tensor(
            module.anchors.numpy(), dtype=tf.float32
        )
        self.anchor_grid = tf.reshape(
            tf.convert_to_tensor(
                module.anchor_grid.numpy(), dtype=tf.float32
            ), [self.nl, 1, -1, 1, 2]
        )
        self.m = [
            tf_Conv2d(
                in_channels=in_channels,
                out_channels=self.no * self.na,
                kernel_size=1,
                module=module.m[i]
            ) for i, in_channels in enumerate(channels)
        ]
        self.export = False  # onnx export
        self.training = True  # set to False after building model
        for i in range(self.nl):
            nx = self.imgsize[1] // self.stride[i]
            ny = self.imgsize[0] // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)
        return

    def call(
        self: tf_Detect,
        inputs: tf.Tensor
    ) -> Union[List[tf.Tensor], Tuple[tf.Tensor, List[tf.Tensor]]]:
        self.training |= self.export
        z = list()  # inference output
        x = list()
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            nx = self.imgsize[1] // self.stride[i]
            ny = self.imgsize[0] // self.stride[i]
            x[i] = tf.transpose(
                tf.reshape(
                    x[i], [-1, ny * nx, self.na, self.no]
                ),
                [0, 2, 1, 3]
            )
            if not self.training:  # inference
                y = tf.sigmoid(x[i])
                xy = (
                    y[..., 0:2] * 2. - 0.5 + self.grid[i]
                ) * self.stride[i]
                wh = (
                    y[..., 2:4] * 2
                ) ** 2 * self.anchor_grid[i]
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant(
                    [[self.imgsize[1], self.imgsize[0]]],
                    dtype=tf.float32
                )
                wh /= tf.constant(
                    [[self.imgsize[1], self.imgsize[0]]],
                    dtype=tf.float32
                )
                y = tf.concat([xy, wh, y[..., 4:]], -1)
                z.append(tf.reshape(y, [-1, 3 * ny * nx, self.no]))
        if self.training:
            return x
        else:
            return tf.concat(z, 1), x

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20):
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(
            tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]),
            dtype=tf.float32
        )


# wrapper of torch.nn.Upsample
class tf_Upsample(Layer):
    def __init__(
        self: tf_Upsample,
        size: Optional[Union[int, Tuple[int, int]]],
        scale_factor: float,
        mode: str,
        module: nn.Module = None
    ) -> None:
        super().__init__()
        assert module is not None
        assert scale_factor == 2, "scale_factor must be 2"
        # upsample = UpSampling2D(size=scale_factor, interpolation=mode)
        self.upsample = lambda x: tf.image.resize(
            x, (x.shape[1] * 2, x.shape[2] * 2), method=mode
        )

    def call(self: tf_Upsample, inputs: tf.Tensor) -> tf.Tensor:
        return self.upsample(inputs)


# wrapper of YOLO V5 Concat Layer (models.common.Concat)
class tf_Concat(Layer):
    def __init__(
        self: tf_Concat,
        dimension: int = 1,
        module: nn.Module = None
    ):
        super().__init__()
        assert module is not None
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3
        return

    def call(self: tf_Concat, inputs: tf.Tensor) -> tf.Tensor:
        return tf.concat(inputs, self.d)


def parse_model(
    config: Dict,
    channels: List[int],
    model_torch: nn.Module
) -> Tuple:
    anchors = config['anchors']
    nc = config['nc']
    gd = config['depth_multiple']
    gw = config['width_multiple']
    # number of anchors
    if isinstance(anchors, list):
        na = len(anchors[0]) // 2
    else:
        na = anchors
    # number of outputs = anchors * (classes + 5)
    no = na * (nc + 5)
    layers = list()
    save = list()
    for i, (from_layer, number, module, args) in enumerate(
            config['backbone'] + config['head']
    ):
        module_str = module
        if isinstance(module, str):
            # eval module
            module = eval(module)
        for j, arg in enumerate(args):
            try:
                if isinstance(arg, str):
                    # eval strings
                    args[j] = eval(arg)
            except Exception:
                pass
        # depth gain
        number = max(round(number * gd), 1) if number > 1 else number
        if module in [
            nn.Conv2d, Conv, Bottleneck, SPP, Focus, BottleneckCSP, C3
        ]:
            in_channels = channels[from_layer]
            out_channels = args[0]
            if out_channels != no:
                out_channels = make_divisible(out_channels * gw, 8)
            args = [in_channels, out_channels, *args[1:]]
            if module in [BottleneckCSP, C3]:
                args.insert(2, number)
                number = 1
        elif module is nn.BatchNorm2d:
            args = [channels[from_layer]]
        elif module is Concat:
            assert isinstance(from_layer, list)
            out_channels = sum([
                channels[-1 if x == -1 else x + 1] for x in from_layer
            ])
        elif module is Detect:
            assert isinstance(from_layer, list)
            args.append([channels[x + 1] for x in from_layer])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [
                    list(range(args[1] * 2))
                ] * len(from_layer)
        else:
            out_channels = channels[from_layer]
        module_tf = eval('tf_' + module_str.replace('nn.', ''))
        if number > 1:
            m_tf_ = Sequential([
                module_tf(*args, module=model_torch.model[i][j])
                for j in range(number)
            ])
            m_torch_ = nn.Sequential(*[
                module(*args) for _ in range(number)
            ])
        else:
            m_tf_ = module_tf(*args, module=model_torch.model[i])
            m_torch_ = module(*args)
        module_type = str(module)[8:-1].replace('__main__.', '')
        num_param = sum([
            x.numel() for x in m_torch_.parameters()
        ])
        # attach index, 'from' index, type, number params
        m_tf_.i = i
        m_tf_.f = from_layer
        m_tf_.type = module_type
        m_tf_.np = num_param
        # append to savelist
        if isinstance(from_layer, int):
            save.extend([
                x % i for x in [from_layer] if x != -1
            ])
        else:
            save.extend([
                x % i for x in from_layer if x != -1
            ])
        layers.append(m_tf_)
        channels.append(out_channels)
    return Sequential(layers), sorted(save)


# wrapper of YOLO V5 Detect Layer (models.yolo.Detect)
class tf_YoloV5(Layer):
    def __init__(
        self: tf_YoloV5,
        model_torch: nn.Module,
        nclasses: int,
        config: Dict
    ) -> None:
        super().__init__()
        self.config = config
        self.config['nc'] = nclasses
        self.model, self.savelist = parse_model(
            config=self.config.copy(),
            channels=[3],
            model_torch=model_torch
        )
        return

    def predict(self: tf_YoloV5, inputs: tf.Tensor) -> tf.Tensor:
        y = list()  # outputs
        x = inputs
        for i, m in enumerate(self.model.layers):
            if m.f != -1:
                # if not from previous layer
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    # from earlier layers
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            if m.i in self.savelist:
                y.append(x)
            else:
                y.append(None)
        return x[0]


class WrapperYoloV5(Layer):
    def __init__(self: WrapperYoloV5, yolov5: tf_YoloV5) -> None:
        super().__init__()
        self.yolov5 = yolov5
        return

    def call(self: WrapperYoloV5, images: tf.Tensor) -> tf.Tensor:
        return self.yolov5(images=images)
