#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import List
from detector.base import Session, Config, Framework, Model, Detector
import os
import numpy as np
import onnxruntime as rt
import torch
import tensorflow as tf
from models.tf_yolov5 import WrapperYoloV5
from openvino.inference_engine import IECore
from utils.convert_tflite import load_frozen_graph

IMAGE_SIZES = {
    'yolov3-tiny': 512,
    'yolov3': 512,
    'yolov3-spp': 512,
    'yolov4-tiny': 512,
    'yolov4': 512,
    'yolov4-csp': 640,
    'yolov4x-mish': 640,
}
STRIDE_ANCHORS = {
    'yolov3-tiny': {
        16: [(10, 14), (23, 27), (37, 58)],
        32: [(81, 82), (135, 169), (344, 319)],
    },
    'yolov3': {
        8: [(10, 13), (16, 30), (33, 23)],
        16: [(30, 61), (62, 45), (59, 119)],
        32: [(116, 90), (156, 198), (373, 326)],
    },
    'yolov3-spp': {
        8: [(10, 13), (16, 30), (33, 23)],
        16: [(30, 61), (62, 45), (59, 119)],
        32: [(116, 90), (156, 198), (373, 326)],
    },
    'yolov4-tiny': {
        16: [(10, 14), (23, 27), (37, 58)],
        32: [(81, 82), (135, 169), (344, 319)],
    },
    'yolov4': {
        8: [(12, 16), (19, 36), (40, 28)],
        16: [(36, 75), (76, 55), (72, 146)],
        32: [(142, 110), (192, 243), (459, 401)],
    },
    'yolov4-csp': {
        8: [(12, 16), (19, 36), (40, 28)],
        16: [(36, 75), (76, 55), (72, 146)],
        32: [(142, 110), (192, 243), (459, 401)],
    },
    'yolov4x-mish': {
        8: [(12, 16), (19, 36), (40, 28)],
        16: [(36, 75), (76, 55), (72, 146)],
        32: [(142, 110), (192, 243), (459, 401)],
    },
}
STRIDE_XYSCALES = {
    'yolov3-tiny': {16: 1.0, 32: 1.0},
    'yolov3': {8: 1.0, 16: 1.0, 32: 1.0},
    'yolov3-spp': {8: 1.0, 16: 1.0, 32: 1.0},
    'yolov4-tiny': {16: 1.05, 32: 1.05},
    'yolov4': {8: 1.05, 16: 1.1, 32: 1.2},
    'yolov4-csp': {8: 2.0, 16: 2.0, 32: 2.0},
    'yolov4x-mish': {8: 2.0, 16: 2.0, 32: 2.0},
}
path_wt = 'weights/yolo'


def sigmoid(x: np.ndarray) -> np.ndarray:
    sigmoid_range = 34.538776394910684
    x = np.clip(x, -sigmoid_range, sigmoid_range)
    return 1.0 / (1.0 + np.exp(-x))


class YoloTFOnnx(Framework):
    def __init__(self: YoloTFOnnx, config: Config) -> None:
        super().__init__(config=config)
        path_model = f'{path_wt}/tf_{config.model}.onnx'
        if not os.path.isfile(path_model):
            raise SystemError(f'onnx({path_model}) not found')
        self.sess = rt.InferenceSession(path_model)
        input_blob = [x.name for x in self.sess.get_inputs()]
        assert len(input_blob) == 1 and input_blob[0] == 'x:0'
        self.input_name = input_blob[0]
        input_shape = self.sess.get_inputs()[0].shape
        assert input_shape[2] == IMAGE_SIZES[self.config.model]
        assert input_shape[3] == IMAGE_SIZES[self.config.model]
        self.output_blob = [x.name for x in self.sess.get_outputs()]
        return

    def inference(self: YoloTFOnnx, sess: Session) -> List[np.ndarray]:
        preds = self.sess.run(
            output_names=self.output_blob,
            input_feed=sess.yolo_input
        )
        preds = [np.squeeze(x, 0).copy() for x in preds]
        return preds


class YoloTFLite(Framework):
    def __init__(self: YoloTFLite, config: Config) -> None:
        super().__init__(config=config)
        path_model = f'{path_wt}/{config.model}_{config.quantize}.tflite'
        if not os.path.isfile(path_model):
            raise SystemError(f'tflite({path_model}) not found')
        self.interpreter = tf.lite.Interpreter(path_model)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        assert input_shape[1] == IMAGE_SIZES[self.config.model]
        assert input_shape[2] == IMAGE_SIZES[self.config.model]
        self.input_name = 'images'
        self.input_index = input_details[0]['index']
        output_details = self.interpreter.get_output_details()
        self.output_indexes = [
            x['index'] for x in output_details
        ]
        if config.quantize == 'int8':
            self.output_quant_params = [
                x['quantization_parameters'] for x in output_details
            ]
        return

    def inference(self: YoloTFLite, sess: Session) -> List[np.ndarray]:
        self.interpreter.set_tensor(
            self.input_index,
            sess.yolo_input[self.input_name]
        )
        self.interpreter.invoke()
        if self.config.quantize == 'int8':
            preds = list()
            for index, params in zip(
                self.output_indexes, self.output_quant_params
            ):
                raw = self.interpreter.get_tensor(index)
                out = (
                    raw.astype(np.float32) - params['zero_points']
                ) * params['scales']
                preds.append(out)
        else:
            preds = [
                self.interpreter.get_tensor(x)
                for x in self.output_indexes
            ]
        preds = [np.squeeze(x, 0).copy() for x in preds]
        return preds


class YoloTF(Framework):
    def __init__(self: YoloTF, config: Config) -> None:
        super().__init__(config=config)
        path_pb = f'{path_wt}/{config.model}.pb'
        if not os.path.isfile(path_pb):
            raise SystemError(f'pb({path_pb}) not found')
        inputs = ['x:0']
        if config.model in ['yolov3-tiny', 'yolov4-tiny']:
            outputs = ['Identity:0', 'Identity_1:0']
        else:
            outputs = ['Identity:0', 'Identity_1:0', 'Identity_2:0']
        self.model = load_frozen_graph(
            path_pb=path_pb,
            inputs=inputs,
            outputs=outputs
        )
        self.input_name = 'images'
        return

    def inference(self: YoloTF, sess: Session) -> List[np.ndarray]:
        preds = self.model(tf.convert_to_tensor(
            sess.yolo_input[self.input_name]
        ))
        preds = [tf.squeeze(x).numpy() for x in preds]
        return preds


class YoloOnnxTF(Framework):
    def __init__(self: YoloOnnxTF, config: Config) -> None:
        super().__init__(config=config)
        path_weight = f'{path_wt}/onnx_tf_{config.model}'
        if not os.path.isdir(path_weight):
            raise SystemError(f'weight({path_weight}) not found')
        model_sm = tf.keras.models.load_model(path_weight)
        self.model = WrapperYoloV5(yolov5=model_sm)
        self.input_name = 'images'
        return

    def inference(self: YoloOnnxTF, sess: Session) -> List[np.ndarray]:
        pred = self.model(sess.yolov5_input[self.input_name])
        return np.squeeze(pred[0].numpy(), 0).copy()


class YoloVino(Framework):
    def __init__(self: YoloVino, config: Config) -> None:
        super().__init__(config=config)
        model = config.model
        if not os.path.isdir(f'{path_wt}/onnx_vino_{model}'):
            raise ValueError(f'OpenVINO IR not found: {model}')
        model_xml = f'{path_wt}/onnx_vino_{model}/{model}.xml'
        model_bin = f'{path_wt}/onnx_vino_{model}/{model}.bin'
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)
        input_blob = list(net.input_info.keys())
        assert len(input_blob) == 1 and input_blob[0] == 'images'
        self.input_name = input_blob[0]
        input_shape = net.input_info[self.input_name].input_data.shape
        assert input_shape[2] == IMAGE_SIZES[self.config.model]
        assert input_shape[3] == IMAGE_SIZES[self.config.model]
        output_blob = list(net.outputs.keys())
        assert 'output' in output_blob
        self.output_blob = ['output']
        self.exec_net = ie.load_network(network=net, device_name='CPU')
        return

    def inference(self: YoloVino, sess: Session) -> List[np.ndarray]:
        pred = self.exec_net.infer(inputs=sess.yolov5_input)
        pred = [pred[ob] for ob in self.output_blob]
        return np.squeeze(pred[0], 0).copy()


class YoloOnnx(Framework):
    def __init__(self: YoloOnnx, config: Config) -> None:
        super().__init__(config=config)
        path_model = f'{path_wt}/{config.model}.onnx'
        if not os.path.isfile(path_model):
            raise SystemError(f'onnx({path_model}) not found')
        self.sess = rt.InferenceSession(path_model)
        input_blob = [x.name for x in self.sess.get_inputs()]
        assert len(input_blob) == 1 and input_blob[0] == 'images'
        self.input_name = input_blob[0]
        input_shape = self.sess.get_inputs()[0].shape
        assert input_shape[2] == IMAGE_SIZES[self.config.model]
        assert input_shape[3] == IMAGE_SIZES[self.config.model]
        output_blob = [x.name for x in self.sess.get_outputs()]
        assert 'output' in output_blob
        self.output_blob = ['output']
        return

    def inference(self: YoloOnnx, sess: Session) -> List[np.ndarray]:
        pred = self.sess.run(
            output_names=self.output_blob,
            input_feed=sess.yolov5_input
        )
        return np.squeeze(pred[0], 0).copy()


class YoloTorch(Framework):
    def __init__(self: YoloTorch, config: Config) -> None:
        super().__init__(config=config)
        path_weight = f'{path_wt}/{config.model}.pt'
        if not os.path.isfile(path_weight):
            raise SystemError(f'weight({path_weight}) not found')
        repo = 'ultralytics/yolov5'
        model = torch.hub.load(repo, config.model, pretrained=False)
        ckpt = torch.load(path_weight, map_location='cpu')['model']
        model.load_state_dict(ckpt.state_dict())
        model.names = ckpt.names
        self.model = model.float().fuse()
        self.model.eval()
        self.input_name = 'images'
        return

    def inference(self: YoloTorch, sess: Session) -> List[np.ndarray]:
        input_feed = torch.from_numpy(
            sess.yolov5_input[self.input_name]
        ).to('cpu')
        with torch.no_grad():
            pred = self.model(input_feed, augment=True)[0]
        return np.squeeze(pred.detach().numpy(), 0).copy()


class Yolo(Model):
    def __init__(self: Yolo, config: Config) -> None:
        super().__init__(config=config)
        if config.framework == 'tf':
            self.framework = YoloTF(config=config)
        elif config.framework == 'tflite':
            self.framework = YoloTFLite(config=config)
        elif config.framework == 'tf_onnx':
            self.framework = YoloTFOnnx(config=config)
        else:
            raise SystemError(
                f'YOLO unsupport {config.framework}'
            )
        return

    def prep_image(self: Yolo, sess: Session) -> None:
        image_size = IMAGE_SIZES[self.config.model]
        sess.padding_image(
            model_height=image_size, model_width=image_size
        )
        image = sess.pad_image
        # reshape image to throw it to the model
        image = image[:, :, ::-1]  # BGR -> RGB
        if self.config.framework in [
            'torch', 'torch_onnx', 'onnx_vino', 'onnx_tf', 'tf_onnx'
        ]:
            image = image.transpose((2, 0, 1))  # HWC -> CHW
        image = image[np.newaxis, ...]
        if (
            self.config.framework == 'tflite'
        ) and (
            self.config.quantize == 'int8'
        ):
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32)
            image /= 255.0
        sess.yolo_input = {self.framework.input_name: image}
        return

    @staticmethod
    def apply_anchors_ver1(
        pred: np.ndarray,
        anchor_grid: np.ndarray,
        stride: int,
        anchor: List[int],
        xyscale: float
    ) -> np.ndarray:
        # xy: min_x, min_y
        # wh: width, height
        # conf: confidence score of the bounding box
        # prob: probability for each category
        xy, wh, conf, prob = np.split(
            pred, (2, 4, 5), axis=-1
        )
        # apply anchor to xy
        xy = ((
            (sigmoid(xy) * xyscale) - (0.5 * (xyscale - 1))
        ) + anchor_grid) * np.array([stride, stride])
        # apply anchor to wh
        wh = np.exp(wh) * np.array(anchor)
        # do sigmoid to probability
        conf = sigmoid(conf)
        prob = sigmoid(prob)
        # concat
        bbox = np.concatenate([xy, wh, conf, prob], axis=-1)
        # expand all anchors
        bbox = np.reshape(bbox, (-1, 85))
        return bbox

    @staticmethod
    def apply_anchors_ver2(
        pred: np.ndarray,
        anchor_grid: np.ndarray,
        stride: int,
        anchor: List[int],
        xyscale: float
    ) -> np.ndarray:
        anchor = (np.array(anchor) / stride)[np.newaxis, np.newaxis, :, :]
        pred = sigmoid(pred)
        # apply anchor to xy
        pred[..., :2] = pred[..., :2] * xyscale - 0.5 + anchor_grid
        # apply anchor to wh
        pred[..., 2:4] = (pred[..., 2:4] * xyscale) ** 2 * anchor
        pred[..., :4] *= stride
        # expand all anchors
        bbox = np.reshape(pred, (-1, 85))
        return bbox

    def apply_anchors(self: Yolo, preds: List[np.ndarray]) -> np.ndarray:
        image_size = IMAGE_SIZES[self.config.model]
        anchors = STRIDE_ANCHORS[self.config.model]
        xyscales = STRIDE_XYSCALES[self.config.model]
        applied = list()
        for pred in preds:
            stride = image_size // max(pred.shape[:2])
            anchor = anchors[stride]
            xyscale = xyscales[stride]
            pred = np.reshape(pred, (*pred.shape[:2], 3, 85))
            # calc grid of anchor box
            anchor_grid = np.meshgrid(
                np.arange(pred.shape[1]), np.arange(pred.shape[0])
            )
            anchor_grid = np.stack(
                anchor_grid, axis=-1
            )[:, :, np.newaxis, :]
            # call each version of apply_anchors
            if self.config.model in ['yolov4-csp', 'yolov4x-mish']:
                bbox = self.apply_anchors_ver2(
                    pred=pred,
                    anchor_grid=anchor_grid,
                    stride=stride,
                    anchor=anchor,
                    xyscale=xyscale
                )
            else:
                bbox = self.apply_anchors_ver1(
                    pred=pred,
                    anchor_grid=anchor_grid,
                    stride=stride,
                    anchor=anchor,
                    xyscale=xyscale
                )
            applied.append(bbox)
        return np.concatenate(applied, axis=0)

    def inference(self: Yolo, sess: Session) -> np.ndarray:
        preds = super().inference(sess=sess)
        pred = self.apply_anchors(preds=preds)
        assert len(pred.shape) == 2
        assert pred.shape[1] == 85
        # xywh -> xyxy
        xywh = pred[:, :4]
        xyxy = np.concatenate([
            (xywh[:, :2] - (xywh[:, 2:] * 0.5)),
            (xywh[:, :2] + (xywh[:, 2:] * 0.5))
        ], axis=-1)
        # rescale bouding boxes according to image preprocessing
        xyxy = sess.rescale_xyxy(xyxy)
        # confidence score of bbox and probability for each category
        conf = pred[:, 4:5]
        prob = pred[:, 5:]
        # confidence score for each category = conf * prob
        cat_conf = conf * prob
        if self.config.model == 'yolov3-tiny':
            cat_conf = np.power(cat_conf, 0.3)
        # catgory of bouding box is the most plausible category
        cat = cat_conf.argmax(axis=1)[:, np.newaxis].astype(np.float)
        # confidence score of bbox is that of the most plausible category
        conf = cat_conf.max(axis=1)[:, np.newaxis]
        # ready for NMS (0-3: xyxy, 4: category id, 5: confidence score)
        return np.concatenate((xyxy, cat, conf), axis=1)


class DetectorYolo(Detector):
    def __init__(self: DetectorYolo, config: Config) -> None:
        super().__init__(config=config)
        self.model = Yolo(config=config)
        return
