#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Conv2D, BatchNormalization,
    ZeroPadding2D, MaxPool2D,
    Add, Lambda
)
from tensorflow_addons.activations import mish


class WeightedLayer(object):
    def __init__(
        self: WeightedLayer,
        stride: int,
        act: str
    ) -> None:
        self.stride = stride
        self.act = act
        self.conv = None
        self.norm = None
        self.input_shape = None
        return

    def __str__(self: WeightedLayer) -> str:
        if self.norm is None:
            bn = 0
        else:
            bn = 1
        rets = list()
        rets.append(f'bn={bn}')
        rets.append(f'filters={self.conv.filters}')
        rets.append(f'size={self.conv.kernel_size[0]}')
        rets.append(f'stride={self.stride}')
        rets.append(f'act={self.act}')
        rets.append(f'in_dim={self.input_shape[-1]}')
        return ', '.join(rets)


class DarknetConv(Layer):
    def __init__(
        self: DarknetConv,
        fil: int,
        ksize: int,
        act: bool = True,  # activation
        actfunc: str = 'leaky',
        ds: bool = False,  # down sampling
        bn: bool = True  # batch normalizaion
    ) -> None:
        super().__init__()
        self.weighted_layers = list()
        # padding
        if ds:
            self.pad = ZeroPadding2D(((1, 0), (1, 0)))
            strides = 2
            padding = 'VALID'
        else:
            self.pad = tf.identity
            strides = 1
            padding = 'SAME'
        # convolution
        self.conv = Conv2D(
            filters=fil,
            kernel_size=ksize,
            strides=strides,
            padding=padding,
            use_bias=not bn,
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            kernel_initializer=tf.random_normal_initializer(
                stddev=0.01
            ),
            bias_initializer=tf.constant_initializer(0.)
        )
        # batch normalization
        if bn:
            self.bn = BatchNormalization()
        else:
            self.bn = tf.identity
        # activation
        if act and actfunc == 'leaky':
            self.act = Lambda(
                lambda x: tf.nn.leaky_relu(x, alpha=0.1)
            )
        elif act and actfunc == 'mish':
            self.act = Lambda(
                lambda x: mish(x)
            )
        else:
            self.act = tf.identity
        self.weighted_layer = WeightedLayer(
            stride=strides,
            act=actfunc if act else 'none'
        )
        self.weighted_layers.append(self.weighted_layer)
        return

    def build(self: DarknetConv, input_shape: Tuple) -> None:
        self.weighted_layer.conv = self.conv
        self.weighted_layer.norm = self.bn
        self.weighted_layer.input_shape = input_shape[1:]
        return

    def call(self: DarknetConv, x: tf.Tensor) -> tf.Tensor:
        return self.act(self.bn(self.conv(self.pad(x))))


class DarknetResidual(Layer):
    def __init__(
        self: DarknetResidual,
        fils: Tuple[int, int],
        actfunc: str = 'leaky'
    ) -> None:
        super().__init__()
        self.weighted_layers = list()
        f1, f2 = fils
        self.dc1 = DarknetConv(fil=f2, ksize=1, actfunc=actfunc)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(fil=f1, ksize=3, actfunc=actfunc)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.add = Add()
        return

    def call(self: DarknetResidual, x: tf.Tensor) -> tf.Tensor:
        short_cut = x
        return self.add([short_cut, self.dc2(self.dc1(x))])


class DarknetBlock(Layer):
    def __init__(
        self: DarknetResidual,
        fils: Tuple[int, int],
        blocks: int
    ) -> None:
        super().__init__()
        self.weighted_layers = list()
        f1, f2 = fils
        self.conv = DarknetConv(fil=f1, ksize=3, ds=True)
        self.weighted_layers.extend(self.conv.weighted_layers)
        self.blocks = tf.keras.Sequential([
            DarknetResidual(fils=fils) for _ in range(blocks)
        ])
        for block in self.blocks.layers:
            self.weighted_layers.extend(block.weighted_layers)
        return

    def call(self: DarknetBlock, x: tf.Tensor) -> tf.Tensor:
        return self.blocks(self.conv(x))


class Darknet53_tiny(Layer):
    def __init__(self: Darknet53_tiny) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc0 = DarknetConv(fil=16, ksize=3)
        self.weighted_layers.extend(self.dc0.weighted_layers)
        self.mp1 = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        self.dc1 = DarknetConv(fil=32, ksize=3)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.mp2 = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        self.dc2 = DarknetConv(fil=64, ksize=3)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.mp3 = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        self.dc3 = DarknetConv(fil=128, ksize=3)
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.mp4 = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        self.dc4 = DarknetConv(fil=256, ksize=3)
        self.weighted_layers.extend(self.dc4.weighted_layers)
        self.mp5 = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        self.dc5 = DarknetConv(fil=512, ksize=3)
        self.weighted_layers.extend(self.dc5.weighted_layers)
        self.mp6 = MaxPool2D(pool_size=2, strides=1, padding='SAME')
        self.dc6 = DarknetConv(fil=1024, ksize=3)
        self.weighted_layers.extend(self.dc6.weighted_layers)
        return

    def call(
        self: Darknet53_tiny,
        x: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.dc4(self.mp4(
            self.dc3(self.mp3(
                self.dc2(self.mp2(
                    self.dc1(self.mp1(self.dc0(x)))
                ))
            ))
        ))
        x_8 = x
        x = self.dc6(self.mp6(self.dc5(self.mp5(x))))
        return x_8, x


class Darknet53(Layer):
    '''YOLOv3
    https://arxiv.org/abs/1804.02767
    '''
    def __init__(self: Darknet53) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc = DarknetConv(fil=32, ksize=3)
        self.weighted_layers.extend(self.dc.weighted_layers)
        self.db1 = DarknetBlock(fils=(64, 32), blocks=1)
        self.weighted_layers.extend(self.db1.weighted_layers)
        self.db2 = DarknetBlock(fils=(128, 64), blocks=2)
        self.weighted_layers.extend(self.db2.weighted_layers)
        self.db3 = DarknetBlock(fils=(256, 128), blocks=8)
        self.weighted_layers.extend(self.db3.weighted_layers)
        self.db4 = DarknetBlock(fils=(512, 256), blocks=8)
        self.weighted_layers.extend(self.db4.weighted_layers)
        self.db5 = DarknetBlock(fils=(1024, 512), blocks=4)
        self.weighted_layers.extend(self.db5.weighted_layers)
        return

    def call(
        self: Darknet53,
        x: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = self.db3(self.db2(self.db1(self.dc(x))))
        x_36 = x
        x = self.db4(x)
        x_61 = x
        x = self.db5(x)
        return x_36, x_61, x


class DarknetBlock_CSPnet(Layer):
    '''Darknet block with CSPnet
    Cross Stage Partial Network
    https://arxiv.org/abs/1911.11929
    '''
    def __init__(
        self: DarknetBlock_CSPnet,
        dcfils: Tuple[int, int],
        drfils: Tuple[int, int],
        blocks: int
    ) -> None:
        super().__init__()
        self.weighted_layers = list()
        f1, f2 = dcfils
        self.dc1 = DarknetConv(fil=f1, ksize=3, ds=True, actfunc='mish')
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(fil=f2, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.dc3 = DarknetConv(fil=f2, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.drs = tf.keras.Sequential([
            DarknetResidual(fils=drfils, actfunc='mish')
            for _ in range(blocks)
        ])
        for dr in self.drs.layers:
            self.weighted_layers.extend(dr.weighted_layers)
        self.dc4 = DarknetConv(fil=f2, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc4.weighted_layers)
        self.dc5 = DarknetConv(fil=f1, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc5.weighted_layers)
        return

    def call(self: DarknetBlock_CSPnet, x: tf.Tensor) -> tf.Tensor:
        route_1 = self.dc1(x)
        route_2 = route_1
        route_2 = self.dc2(route_2)
        route_1 = self.dc4(self.drs(self.dc3(route_1)))
        x = tf.concat([route_1, route_2], axis=-1)
        return self.dc5(x)


class SPP(Layer):
    '''SPP block
    Spatial Pyramid Pooling
    https://arxiv.org/abs/1406.4729
    '''
    def __init__(self: SPP) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc1 = DarknetConv(fil=512, ksize=1)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(fil=1024, ksize=3)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.dc3 = DarknetConv(fil=512, ksize=1)
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.mp1 = MaxPool2D(pool_size=13, strides=1, padding='SAME')
        self.mp2 = MaxPool2D(pool_size=9, strides=1, padding='SAME')
        self.mp3 = MaxPool2D(pool_size=5, strides=1, padding='SAME')
        self.dc4 = DarknetConv(fil=512, ksize=1)
        self.weighted_layers.extend(self.dc4.weighted_layers)
        self.dc5 = DarknetConv(fil=1024, ksize=3)
        self.weighted_layers.extend(self.dc5.weighted_layers)
        self.dc6 = DarknetConv(fil=512, ksize=1)
        self.weighted_layers.extend(self.dc6.weighted_layers)
        return

    def call(self: SPP, x: tf.Tensor) -> tf.Tensor:
        x = self.dc3(self.dc2(self.dc1(x)))
        x = tf.concat([
            self.mp1(x), self.mp2(x), self.mp3(x), x
        ], axis=-1)
        return self.dc6(self.dc5(self.dc4(x)))


class Darknet53_CSPnet(Layer):
    '''YOLOv4
    https://arxiv.org/abs/2004.10934
    '''
    def __init__(self: Darknet53_CSPnet) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc = DarknetConv(fil=32, ksize=3, actfunc='mish')
        self.weighted_layers.extend(self.dc.weighted_layers)
        self.db1 = DarknetBlock_CSPnet(
            dcfils=(64, 64), drfils=(64, 32), blocks=1
        )
        self.weighted_layers.extend(self.db1.weighted_layers)
        self.db2 = DarknetBlock_CSPnet(
            dcfils=(128, 64), drfils=(64, 64), blocks=2
        )
        self.weighted_layers.extend(self.db2.weighted_layers)
        self.db3 = DarknetBlock_CSPnet(
            dcfils=(256, 128), drfils=(128, 128), blocks=8
        )
        self.weighted_layers.extend(self.db3.weighted_layers)
        self.db4 = DarknetBlock_CSPnet(
            dcfils=(512, 256), drfils=(256, 256), blocks=8
        )
        self.weighted_layers.extend(self.db4.weighted_layers)
        self.db5 = DarknetBlock_CSPnet(
            dcfils=(1024, 512), drfils=(512, 512), blocks=4
        )
        self.weighted_layers.extend(self.db5.weighted_layers)
        self.spp = SPP()
        self.weighted_layers.extend(self.spp.weighted_layers)
        return

    def call(
        self: Darknet53_CSPnet,
        x: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = self.db3(self.db2(self.db1(self.dc(x))))
        x_54 = x
        x = self.db4(x)
        x_85 = x
        x = self.spp(self.db5(x))
        return x_54, x_85, x


class YoloLayer(Layer):
    def __init__(self: YoloLayer, fil: int, nc: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        fil2 = 3 * (nc + 5)
        self.dc1 = DarknetConv(fil=fil, ksize=3)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(fil=fil2, ksize=1, act=False, bn=False)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        return

    def call(self: YoloLayer, x: tf.Tensor) -> tf.Tensor:
        return self.dc2(self.dc1(x))


class UpSampling(Layer):
    def __init__(self: UpSampling, fil: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc = DarknetConv(fil=fil, ksize=1)
        self.weighted_layers.extend(self.dc.weighted_layers)
        return

    def call(self: UpSampling, x: tf.Tensor) -> tf.Tensor:
        double_shape = (x.shape[1] * 2, x.shape[2] * 2)
        return tf.image.resize(self.dc(x), double_shape, method='nearest')


class tf_YoloV3_tiny(tf.keras.Model):
    def __init__(self: tf_YoloV3_tiny, nc: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.net = Darknet53_tiny()
        self.weighted_layers.extend(self.net.weighted_layers)
        self.dc = DarknetConv(fil=256, ksize=1)
        self.weighted_layers.extend(self.dc.weighted_layers)
        self.yl1 = YoloLayer(fil=512, nc=nc)
        self.weighted_layers.extend(self.yl1.weighted_layers)
        self.us = UpSampling(fil=128)
        self.weighted_layers.extend(self.us.weighted_layers)
        self.yl2 = YoloLayer(fil=256, nc=nc)
        self.weighted_layers.extend(self.yl2.weighted_layers)
        return

    def call(self: tf_YoloV3_tiny, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        x_8, x = self.net(inputs)
        x = self.dc(x)
        pred = x
        large_bbox = self.yl1(pred)
        pred = tf.concat([self.us(x), x_8], axis=-1)
        middle_bbox = self.yl2(pred)
        return middle_bbox, large_bbox


class DarknetConvSeq(Layer):
    def __init__(self: DarknetConvSeq, fil: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        dcs = list()
        dcs.append(DarknetConv(fil=fil, ksize=1))
        dcs.append(DarknetConv(fil=fil * 2, ksize=3))
        dcs.append(DarknetConv(fil=fil, ksize=1))
        dcs.append(DarknetConv(fil=fil * 2, ksize=3))
        dcs.append(DarknetConv(fil=fil, ksize=1))
        self.dcs = tf.keras.Sequential(dcs)
        for dc in self.dcs.layers:
            self.weighted_layers.extend(dc.weighted_layers)
        return

    def call(self: DarknetConvSeq, x: tf.Tensor) -> tf.Tensor:
        return self.dcs(x)


class tf_YoloV3(tf.keras.Model):
    def __init__(self: tf_YoloV3, nc: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.net = Darknet53()
        self.weighted_layers.extend(self.net.weighted_layers)
        self.dc1 = DarknetConvSeq(fil=512)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.yl1 = YoloLayer(fil=1024, nc=nc)
        self.weighted_layers.extend(self.yl1.weighted_layers)
        self.us1 = UpSampling(fil=256)
        self.weighted_layers.extend(self.us1.weighted_layers)
        self.dc2 = DarknetConvSeq(fil=256)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.yl2 = YoloLayer(fil=512, nc=nc)
        self.weighted_layers.extend(self.yl2.weighted_layers)
        self.us2 = UpSampling(fil=128)
        self.weighted_layers.extend(self.us2.weighted_layers)
        self.dc3 = DarknetConvSeq(fil=128)
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.yl3 = YoloLayer(fil=256, nc=nc)
        self.weighted_layers.extend(self.yl3.weighted_layers)
        return

    def call(self: tf_YoloV3, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        x_36, x_61, x = self.net(inputs)
        x = self.dc1(x)
        pred = x
        large_bbox = self.yl1(pred)
        x = self.dc2(tf.concat([self.us1(x), x_61], axis=-1))
        pred = x
        middle_bbox = self.yl2(pred)
        pred = self.dc3(tf.concat([self.us2(x), x_36], axis=-1))
        small_bbox = self.yl3(pred)
        return small_bbox, middle_bbox, large_bbox


class tf_YoloV4(tf.keras.Model):
    def __init__(self: tf_YoloV4, nc: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.net = Darknet53_CSPnet()
        self.weighted_layers.extend(self.net.weighted_layers)
        self.us1 = UpSampling(fil=256)
        self.weighted_layers.extend(self.us1.weighted_layers)
        self.dc1 = DarknetConv(fil=256, ksize=1)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dcs1 = DarknetConvSeq(fil=256)
        self.weighted_layers.extend(self.dcs1.weighted_layers)
        self.us2 = UpSampling(fil=128)
        self.weighted_layers.extend(self.us2.weighted_layers)
        self.dc2 = DarknetConv(fil=128, ksize=1)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.dcs2 = DarknetConvSeq(fil=128)
        self.weighted_layers.extend(self.dcs2.weighted_layers)
        self.yl1 = YoloLayer(fil=256, nc=nc)
        self.weighted_layers.extend(self.yl1.weighted_layers)
        self.dc3 = DarknetConv(fil=256, ksize=3, ds=True)
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.dcs3 = DarknetConvSeq(fil=256)
        self.weighted_layers.extend(self.dcs3.weighted_layers)
        self.yl2 = YoloLayer(fil=512, nc=nc)
        self.weighted_layers.extend(self.yl2.weighted_layers)
        self.dc4 = DarknetConv(fil=512, ksize=3, ds=True)
        self.weighted_layers.extend(self.dc4.weighted_layers)
        self.dcs4 = DarknetConvSeq(fil=512)
        self.weighted_layers.extend(self.dcs4.weighted_layers)
        self.yl3 = YoloLayer(fil=1024, nc=nc)
        self.weighted_layers.extend(self.yl3.weighted_layers)
        return

    def call(self: tf_YoloV4, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        x_54, x_85, x = self.net(inputs)
        short_cut = x
        x = self.dcs1(tf.concat([self.dc1(x_85), self.us1(x)], axis=-1))
        x_85 = x
        x = self.dcs2(tf.concat([self.dc2(x_54), self.us2(x)], axis=-1))
        pred = x
        small_bbox = self.yl1(pred)
        x = self.dcs3(tf.concat([self.dc3(x), x_85], axis=-1))
        pred = x
        middle_bbox = self.yl2(pred)
        pred = self.dcs4(tf.concat([self.dc4(x), short_cut], axis=-1))
        large_bbox = self.yl3(pred)
        return small_bbox, middle_bbox, large_bbox
