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
from tensorflow.keras.activations import sigmoid
from tensorflow_addons.activations import mish

# ### COMMON LAYERS ###


class WeightedLayer(object):
    '''class for copying darknet weights
    '''
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
        rets = list()
        if self.norm.__class__.__name__ == 'BatchNormalization':
            bn = 1
        else:
            bn = 0
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
        elif act and actfunc == 'sigmoid':
            self.act = Lambda(
                lambda x: sigmoid(x)
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


class UpSampling(Layer):
    def __init__(
        self: UpSampling,
        fil: int,
        actfunc: str = 'leaky'
    ) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc = DarknetConv(fil=fil, ksize=1, actfunc=actfunc)
        self.weighted_layers.extend(self.dc.weighted_layers)
        return

    def call(self: UpSampling, x: tf.Tensor) -> tf.Tensor:
        double_shape = (x.shape[1] * 2, x.shape[2] * 2)
        return tf.image.resize(self.dc(x), double_shape, method='nearest')


class YoloLayer(Layer):
    def __init__(
        self: YoloLayer,
        fil: int,
        nc: int,
        act: bool = False
    ) -> None:
        super().__init__()
        self.weighted_layers = list()
        fil2 = 3 * (nc + 5)
        if act:
            actfunc = 'mish'
        else:
            actfunc = 'leaky'
        self.dc1 = DarknetConv(fil=fil, ksize=3, actfunc=actfunc)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(
            fil=fil2, ksize=1, act=act, actfunc='sigmoid', bn=False
        )
        self.weighted_layers.extend(self.dc2.weighted_layers)
        return

    def call(self: YoloLayer, x: tf.Tensor) -> tf.Tensor:
        return self.dc2(self.dc1(x))

# ### BLOCKS ###


class DarknetResidual(Layer):
    '''Residual Block
    https://arxiv.org/abs/1512.03385
    '''
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
    '''Basic Block for YOLO (stack of Residual Block)
    '''
    def __init__(
        self: DarknetResidual,
        fils: Tuple[int, int],
        blocks: int,
        actfunc: str = 'leaky'
    ) -> None:
        super().__init__()
        self.weighted_layers = list()
        f1, f2 = fils
        self.conv = DarknetConv(fil=f1, ksize=3, actfunc=actfunc, ds=True)
        self.weighted_layers.extend(self.conv.weighted_layers)
        self.blocks = tf.keras.Sequential([
            DarknetResidual(fils=fils, actfunc=actfunc)
            for _ in range(blocks)
        ])
        for block in self.blocks.layers:
            self.weighted_layers.extend(block.weighted_layers)
        return

    def call(self: DarknetBlock, x: tf.Tensor) -> tf.Tensor:
        return self.blocks(self.conv(x))


class DarknetBlock_CSPnet_tiny(Layer):
    '''tiny CSPnet Block for YoloV4_tiny
    '''
    def __init__(self: DarknetBlock_CSPnet_tiny, fil: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        hidden_fil = fil // 2
        self.dc1 = DarknetConv(fil=fil, ksize=3)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(fil=hidden_fil, ksize=3)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.dc3 = DarknetConv(fil=hidden_fil, ksize=3)
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.dc4 = DarknetConv(fil=fil, ksize=1)
        self.weighted_layers.extend(self.dc4.weighted_layers)
        self.mp = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        return

    def call(
        self: DarknetBlock_CSPnet_tiny,
        x: tf.Tensor
    ) -> Tuple[tf.Tensor]:
        x = self.dc1(x)
        route_2 = x
        x = tf.split(x, num_or_size_splits=2, axis=-1)[1]
        x = self.dc2(x)
        route_1 = x
        x = self.dc3(x)
        x = tf.concat([x, route_1], axis=-1)
        x = self.dc4(x)
        route_3 = x
        x = tf.concat([route_2, x], axis=-1)
        x = self.mp(x)
        return route_3, x


class DarknetBlock_CSPnet(Layer):
    '''CSPnet Block for YoloV4
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


class SPP_CSPnet(Layer):
    '''SPP block with CSPnet
    '''
    def __init__(self: SPP_CSPnet, fil: int, blocks: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc1 = DarknetConv(fil=fil, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(fil=fil, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.dc3 = DarknetConv(fil=fil, ksize=3, actfunc='mish')
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.dc4 = DarknetConv(fil=fil, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc4.weighted_layers)
        self.mp1 = MaxPool2D(pool_size=13, strides=1, padding='SAME')
        self.mp2 = MaxPool2D(pool_size=9, strides=1, padding='SAME')
        self.mp3 = MaxPool2D(pool_size=5, strides=1, padding='SAME')
        dcs = list()
        for _ in range(blocks):
            dcs.append(DarknetConv(fil=fil, ksize=1, actfunc='mish'))
            dcs.append(DarknetConv(fil=fil, ksize=3, actfunc='mish'))
        self.dcs = tf.keras.Sequential(dcs)
        for dc in self.dcs.layers:
            self.weighted_layers.extend(dc.weighted_layers)
        self.dc5 = DarknetConv(fil=fil, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc5.weighted_layers)
        return

    def call(self: SPP_CSPnet, x: tf.Tensor) -> tf.Tensor:
        x = self.dc1(x)
        route = x
        x = self.dc4(self.dc3(self.dc2(x)))
        x = self.dcs(tf.concat([
            self.mp1(x), self.mp2(x), self.mp3(x), x
        ], axis=-1))
        x = tf.concat([x, route], axis=-1)
        x = self.dc5(x)
        return x


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


class DarknetConvSeq_CSPnet(Layer):
    def __init__(self: DarknetConvSeq, fil: int, blocks: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc1 = DarknetConv(fil=fil, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(fil=fil, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc2.weighted_layers)
        dcs = list()
        for _ in range(blocks):
            dcs.append(DarknetConv(fil=fil, ksize=1, actfunc='mish'))
            dcs.append(DarknetConv(fil=fil, ksize=3, actfunc='mish'))
        self.dcs = tf.keras.Sequential(dcs)
        for dc in self.dcs.layers:
            self.weighted_layers.extend(dc.weighted_layers)
        self.dc3 = DarknetConv(fil=fil, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc3.weighted_layers)
        return

    def call(self: DarknetConvSeq, x: tf.Tensor) -> tf.Tensor:
        x = self.dc1(x)
        route = x
        x = self.dc3(tf.concat([
            self.dcs(x),
            self.dc2(route)
        ], axis=-1))
        return x

# ### BACKBONES ###


class Darknet53_tiny(Layer):
    '''backbone of YoloV3_tiny
    '''
    def __init__(self: Darknet53_tiny) -> None:
        super().__init__()
        self.weighted_layers = list()
        seq1 = list()
        dc = DarknetConv(fil=16, ksize=3)
        seq1.append(dc)
        self.weighted_layers.extend(dc.weighted_layers)
        mp = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        seq1.append(mp)
        dc = DarknetConv(fil=32, ksize=3)
        seq1.append(dc)
        self.weighted_layers.extend(dc.weighted_layers)
        mp = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        seq1.append(mp)
        dc = DarknetConv(fil=64, ksize=3)
        seq1.append(dc)
        self.weighted_layers.extend(dc.weighted_layers)
        mp = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        seq1.append(mp)
        dc = DarknetConv(fil=128, ksize=3)
        seq1.append(dc)
        self.weighted_layers.extend(dc.weighted_layers)
        mp = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        seq1.append(mp)
        dc = DarknetConv(fil=256, ksize=3)
        seq1.append(dc)
        self.weighted_layers.extend(dc.weighted_layers)
        self.seq1 = tf.keras.Sequential(seq1)
        seq2 = list()
        mp = MaxPool2D(pool_size=2, strides=2, padding='SAME')
        seq2.append(mp)
        dc = DarknetConv(fil=512, ksize=3)
        seq2.append(dc)
        self.weighted_layers.extend(dc.weighted_layers)
        mp = MaxPool2D(pool_size=2, strides=1, padding='SAME')
        seq2.append(mp)
        dc = DarknetConv(fil=1024, ksize=3)
        seq2.append(dc)
        self.weighted_layers.extend(dc.weighted_layers)
        self.seq2 = tf.keras.Sequential(seq2)
        return

    def call(self: Darknet53_tiny, x: tf.Tensor) -> Tuple[tf.Tensor]:
        x = self.seq1(x)
        x_8 = x
        x = self.seq2(x)
        return x_8, x


class Darknet53(Layer):
    '''backbone of YOLOv3 and YOLOv3_spp
    '''
    def __init__(self: Darknet53, spp: bool = False) -> None:
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
        self.is_spp = spp
        if spp:
            self.spp = SPP()
            self.weighted_layers.extend(self.spp.weighted_layers)
        return

    def call(self: Darknet53, x: tf.Tensor) -> Tuple[tf.Tensor]:
        x = self.db3(self.db2(self.db1(self.dc(x))))
        x_36 = x
        x = self.db4(x)
        x_61 = x
        x = self.db5(x)
        if self.is_spp:
            x = self.spp(x)
        return x_36, x_61, x


class Darknet53_CSPnet_tiny(Layer):
    '''backbone of YOLOv4 tiny
    '''
    def __init__(self: Darknet53_CSPnet_tiny) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc1 = DarknetConv(fil=32, ksize=3, ds=True)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dc2 = DarknetConv(fil=64, ksize=3, ds=True)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.db1 = DarknetBlock_CSPnet_tiny(fil=64)
        self.weighted_layers.extend(self.db1.weighted_layers)
        self.db2 = DarknetBlock_CSPnet_tiny(fil=128)
        self.weighted_layers.extend(self.db2.weighted_layers)
        self.db3 = DarknetBlock_CSPnet_tiny(fil=256)
        self.weighted_layers.extend(self.db3.weighted_layers)
        self.dc3 = DarknetConv(fil=512, ksize=3)
        self.weighted_layers.extend(self.dc3.weighted_layers)
        return

    def call(
        self: Darknet53_CSPnet_tiny,
        x: tf.Tensor
    ) -> Tuple[tf.Tensor]:
        x = self.dc2(self.dc1(x))
        _, x = self.db1(x)
        _, x = self.db2(x)
        route, x = self.db3(x)
        x = self.dc3(x)
        return route, x


class Darknet53_CSPnet(Layer):
    '''backbone of YOLOv4
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

    def call(self: Darknet53_CSPnet, x: tf.Tensor) -> Tuple[tf.Tensor]:
        x = self.db3(self.db2(self.db1(self.dc(x))))
        x_54 = x
        x = self.db4(x)
        x_85 = x
        x = self.spp(self.db5(x))
        return x_54, x_85, x


class Darknet53_CSPnet_2(Layer):
    '''backbone of YOLOv4-csp and YOLOv4x-mish
    '''
    def __init__(
        self: Darknet53_CSPnet_2,
        fil: int,
        blocks: Tuple[int]
    ) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.dc = DarknetConv(fil=32, ksize=3, actfunc='mish')
        self.weighted_layers.extend(self.dc.weighted_layers)
        fil1 = fil
        fil2 = fil1 * 2
        self.db1 = DarknetBlock(
            fils=(fil2, fil1), blocks=blocks[0], actfunc='mish'
        )
        self.weighted_layers.extend(self.db1.weighted_layers)
        fil1 *= 2
        fil2 *= 2
        self.db2 = DarknetBlock_CSPnet(
            dcfils=(fil2, fil1), drfils=(fil1, fil1), blocks=blocks[1]
        )
        self.weighted_layers.extend(self.db2.weighted_layers)
        fil1 *= 2
        fil2 *= 2
        self.db3 = DarknetBlock_CSPnet(
            dcfils=(fil2, fil1), drfils=(fil1, fil1), blocks=blocks[2]
        )
        self.weighted_layers.extend(self.db3.weighted_layers)
        fil1 *= 2
        fil2 *= 2
        self.db4 = DarknetBlock_CSPnet(
            dcfils=(fil2, fil1), drfils=(fil1, fil1), blocks=blocks[3]
        )
        self.weighted_layers.extend(self.db4.weighted_layers)
        fil1 *= 2
        fil2 *= 2
        self.db5 = DarknetBlock_CSPnet(
            dcfils=(fil2, fil1), drfils=(fil1, fil1), blocks=blocks[4]
        )
        self.weighted_layers.extend(self.db5.weighted_layers)
        self.spp = SPP_CSPnet(fil=fil1, blocks=blocks[5])
        self.weighted_layers.extend(self.spp.weighted_layers)
        return

    def call(self: Darknet53_CSPnet, x: tf.Tensor) -> Tuple[tf.Tensor]:
        x = self.db3(self.db2(self.db1(self.dc(x))))
        route_1 = x
        x = self.db4(x)
        route_2 = x
        x = self.spp(self.db5(x))
        return route_1, route_2, x

# ### YOLO MODELS ###


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
        x_13 = x
        large_bbox = self.yl1(x)
        x = tf.concat([
            self.us(x_13),
            x_8
        ], axis=-1)
        middle_bbox = self.yl2(x)
        return middle_bbox, large_bbox


class tf_YoloV3(tf.keras.Model):
    '''https://arxiv.org/abs/1804.02767
    '''
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
        x_79 = x
        large_bbox = self.yl1(x)
        x = self.dc2(tf.concat([
            self.us1(x_79),
            x_61
        ], axis=-1))
        x_91 = x
        middle_bbox = self.yl2(x)
        x = self.dc3(tf.concat([
            self.us2(x_91),
            x_36
        ], axis=-1))
        small_bbox = self.yl3(x)
        return small_bbox, middle_bbox, large_bbox


class tf_YoloV3_spp(tf.keras.Model):
    '''https://arxiv.org/abs/1804.02767
    '''
    def __init__(self: tf_YoloV3_spp, nc: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.net = Darknet53(spp=True)
        self.weighted_layers.extend(self.net.weighted_layers)
        self.yl1 = YoloLayer(fil=1024, nc=nc)
        self.weighted_layers.extend(self.yl1.weighted_layers)
        self.us1 = UpSampling(fil=256)
        self.weighted_layers.extend(self.us1.weighted_layers)
        self.dc1 = DarknetConvSeq(fil=256)
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.yl2 = YoloLayer(fil=512, nc=nc)
        self.weighted_layers.extend(self.yl2.weighted_layers)
        self.us2 = UpSampling(fil=128)
        self.weighted_layers.extend(self.us2.weighted_layers)
        self.dc2 = DarknetConvSeq(fil=128)
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.yl3 = YoloLayer(fil=256, nc=nc)
        self.weighted_layers.extend(self.yl3.weighted_layers)
        return

    def call(self: tf_YoloV3_spp, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        x_36, x_61, x = self.net(inputs)
        x_86 = x
        large_bbox = self.yl1(x)
        x = self.dc1(tf.concat([
            self.us1(x_86),
            x_61
        ], axis=-1))
        x_98 = x
        middle_bbox = self.yl2(x)
        x = self.dc2(tf.concat([
            self.us2(x_98),
            x_36
        ], axis=-1))
        small_bbox = self.yl3(x)
        return small_bbox, middle_bbox, large_bbox


class tf_YoloV4_tiny(tf.keras.Model):
    def __init__(self: tf_YoloV4_tiny, nc: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.net = Darknet53_CSPnet_tiny()
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

    def call(self: tf_YoloV3, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        x_23, x = self.net(inputs)
        x = self.dc(x)
        x_27 = x
        large_bbox = self.yl1(x)
        x = tf.concat([self.us(x_27), x_23], axis=-1)
        middle_bbox = self.yl2(x)
        return middle_bbox, large_bbox


class tf_YoloV4(tf.keras.Model):
    '''https://arxiv.org/abs/2004.10934
    '''
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
        x_116 = x
        x = self.dcs1(tf.concat([
            self.dc1(x_85),
            self.us1(x)
        ], axis=-1))
        x_126 = x
        x = self.dcs2(tf.concat([
            self.dc2(x_54),
            self.us2(x)
        ], axis=-1))
        x_136 = x
        large_bbox = self.yl1(x)
        x = self.dcs3(tf.concat([
            self.dc3(x_136),
            x_126
        ], axis=-1))
        x_148 = x
        middle_bbox = self.yl2(x)
        x = self.dcs4(tf.concat([
            self.dc4(x_148),
            x_116
        ], axis=-1))
        small_bbox = self.yl3(x)
        return small_bbox, middle_bbox, large_bbox


class tf_YoloV4_csp(tf.keras.Model):
    '''https://arxiv.org/abs/2011.08036
    '''
    def __init__(self: tf_YoloV4_csp, nc: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.net = Darknet53_CSPnet_2(
            fil=32, blocks=(1, 2, 8, 8, 4, 1)
        )
        self.weighted_layers.extend(self.net.weighted_layers)
        self.us1 = UpSampling(fil=256, actfunc='mish')
        self.weighted_layers.extend(self.us1.weighted_layers)
        self.dc1 = DarknetConv(fil=256, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dcs1 = DarknetConvSeq_CSPnet(fil=256, blocks=2)
        self.weighted_layers.extend(self.dcs1.weighted_layers)
        self.us2 = UpSampling(fil=128, actfunc='mish')
        self.weighted_layers.extend(self.us2.weighted_layers)
        self.dc2 = DarknetConv(fil=128, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.dcs2 = DarknetConvSeq_CSPnet(fil=128, blocks=2)
        self.weighted_layers.extend(self.dcs2.weighted_layers)
        self.yl1 = YoloLayer(fil=256, nc=nc, act=True)
        self.weighted_layers.extend(self.yl1.weighted_layers)
        self.dc3 = DarknetConv(fil=256, ksize=3, ds=True, actfunc='mish')
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.dcs3 = DarknetConvSeq_CSPnet(fil=256, blocks=2)
        self.weighted_layers.extend(self.dcs3.weighted_layers)
        self.yl2 = YoloLayer(fil=512, nc=nc, act=True)
        self.weighted_layers.extend(self.yl2.weighted_layers)
        self.dc4 = DarknetConv(fil=512, ksize=3, ds=True, actfunc='mish')
        self.weighted_layers.extend(self.dc4.weighted_layers)
        self.dcs4 = DarknetConvSeq_CSPnet(fil=512, blocks=2)
        self.weighted_layers.extend(self.dcs4.weighted_layers)
        self.yl3 = YoloLayer(fil=1024, nc=nc, act=True)
        self.weighted_layers.extend(self.yl3.weighted_layers)
        return

    def call(self: tf_YoloV4_csp, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        x_48, x_79, x = self.net(inputs)
        x_113 = x
        x = self.dcs1(tf.concat([
            self.dc1(x_79),
            self.us1(x)
        ], axis=-1))
        x_127 = x
        x = self.dcs2(tf.concat([
            self.dc2(x_48),
            self.us2(x)
        ], axis=-1))
        x_141 = x
        large_bbox = self.yl1(x)
        x = self.dcs3(tf.concat([
            self.dc3(x_141),
            x_127
        ], axis=-1))
        x_156 = x
        middle_bbox = self.yl2(x)
        x = self.dcs4(tf.concat([
            self.dc4(x_156),
            x_113
        ], axis=-1))
        small_bbox = self.yl3(x)
        return small_bbox, middle_bbox, large_bbox


class tf_YoloV4x_mish(tf.keras.Model):
    def __init__(self: tf_YoloV4x_mish, nc: int) -> None:
        super().__init__()
        self.weighted_layers = list()
        self.net = Darknet53_CSPnet_2(
            fil=40, blocks=(1, 3, 10, 10, 5, 2)
        )
        self.weighted_layers.extend(self.net.weighted_layers)
        self.us1 = UpSampling(fil=320, actfunc='mish')
        self.weighted_layers.extend(self.us1.weighted_layers)
        self.dc1 = DarknetConv(fil=320, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc1.weighted_layers)
        self.dcs1 = DarknetConvSeq_CSPnet(fil=320, blocks=3)
        self.weighted_layers.extend(self.dcs1.weighted_layers)
        self.us2 = UpSampling(fil=160, actfunc='mish')
        self.weighted_layers.extend(self.us2.weighted_layers)
        self.dc2 = DarknetConv(fil=160, ksize=1, actfunc='mish')
        self.weighted_layers.extend(self.dc2.weighted_layers)
        self.dcs2 = DarknetConvSeq_CSPnet(fil=160, blocks=3)
        self.weighted_layers.extend(self.dcs2.weighted_layers)
        self.yl1 = YoloLayer(fil=320, nc=nc, act=True)
        self.weighted_layers.extend(self.yl1.weighted_layers)
        self.dc3 = DarknetConv(fil=320, ksize=3, ds=True, actfunc='mish')
        self.weighted_layers.extend(self.dc3.weighted_layers)
        self.dcs3 = DarknetConvSeq_CSPnet(fil=320, blocks=3)
        self.weighted_layers.extend(self.dcs3.weighted_layers)
        self.yl2 = YoloLayer(fil=640, nc=nc, act=True)
        self.weighted_layers.extend(self.yl2.weighted_layers)
        self.dc4 = DarknetConv(fil=640, ksize=3, ds=True, actfunc='mish')
        self.weighted_layers.extend(self.dc4.weighted_layers)
        self.dcs4 = DarknetConvSeq_CSPnet(fil=640, blocks=3)
        self.weighted_layers.extend(self.dcs4.weighted_layers)
        self.yl3 = YoloLayer(fil=1280, nc=nc, act=True)
        self.weighted_layers.extend(self.yl3.weighted_layers)
        return

    def call(self: tf_YoloV4x_mish, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        x_57, x_94, x = self.net(inputs)
        x_133 = x
        x = self.dcs1(tf.concat([
            self.dc1(x_94),
            self.us1(x)
        ], axis=-1))
        x_149 = x
        x = self.dcs2(tf.concat([
            self.dc2(x_57),
            self.us2(x)
        ], axis=-1))
        x_165 = x
        large_bbox = self.yl1(x)
        x = self.dcs3(tf.concat([
            self.dc3(x_165),
            x_149
        ], axis=-1))
        x_182 = x
        middle_bbox = self.yl2(x)
        x = self.dcs4(tf.concat([
            self.dc4(x_182),
            x_133
        ], axis=-1))
        small_bbox = self.yl3(x)
        return small_bbox, middle_bbox, large_bbox
