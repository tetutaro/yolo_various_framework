#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from detector.base import Session, Config, Framework, Model, Detector
import os
import numpy as np
import onnxruntime as rt
import torch
import tensorflow as tf
from models.tf_yolov5 import WrapperYoloV5
from openvino.inference_engine import IECore
from utils.convert_tflite import load_frozen_graph

IMAGE_SIZE = 640
path_wt = 'weights/yolov5'


class YoloV5TFOnnx(Framework):
    def __init__(self: YoloV5TFOnnx, config: Config) -> None:
        super().__init__(config=config)
        path_model = f'{path_wt}/tf_{config.model}.onnx'
        if not os.path.isfile(path_model):
            raise SystemError(f'onnx({path_model}) not found')
        self.sess = rt.InferenceSession(path_model)
        input_blob = [x.name for x in self.sess.get_inputs()]
        assert len(input_blob) == 1 and input_blob[0] == 'x:0'
        self.input_name = input_blob[0]
        input_shape = self.sess.get_inputs()[0].shape
        assert input_shape[2] == IMAGE_SIZE
        assert input_shape[3] == IMAGE_SIZE
        output_blob = [x.name for x in self.sess.get_outputs()]
        assert 'Identity:0' in output_blob
        self.output_blob = ['Identity:0']
        return

    def inference(self: YoloV5TFOnnx, sess: Session) -> np.ndarray:
        pred = self.sess.run(
            output_names=self.output_blob,
            input_feed=sess.yolov5_input
        )
        pred = np.squeeze(pred[0], 0).copy()
        pred[:, :4] = pred[:, :4] * IMAGE_SIZE
        return pred


class YoloV5TFLite(Framework):
    def __init__(self: YoloV5TFLite, config: Config) -> None:
        super().__init__(config=config)
        path_model = f'{path_wt}/{config.model}_{config.quantize}.tflite'
        if not os.path.isfile(path_model):
            raise SystemError(f'tflite({path_model}) not found')
        self.interpreter = tf.lite.Interpreter(path_model)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        assert input_shape[1] == IMAGE_SIZE
        assert input_shape[2] == IMAGE_SIZE
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

    def inference(self: YoloV5TFLite, sess: Session) -> np.ndarray:
        self.interpreter.set_tensor(
            self.input_index,
            sess.yolov5_input[self.input_name]
        )
        self.interpreter.invoke()
        if self.config.quantize == 'int8':
            pred = list()
            for index, params in zip(
                self.output_indexes, self.output_quant_params
            ):
                raw = self.interpreter.get_tensor(index)
                out = (
                    raw.astype(np.float32) - params['zero_points']
                ) * params['scales']
                pred.append(out)
        else:
            pred = [
                self.interpreter.get_tensor(x)
                for x in self.output_indexes
            ]
        pred = np.squeeze(pred[0], 0).copy()
        pred[:, :4] = pred[:, :4] * IMAGE_SIZE
        return pred


class YoloV5TF(Framework):
    def __init__(self: YoloV5TF, config: Config) -> None:
        super().__init__(config=config)
        path_pb = f'{path_wt}/{config.model}.pb'
        if not os.path.isfile(path_pb):
            raise SystemError(f'pb({path_pb}) not found')
        self.model = load_frozen_graph(
            path_pb=path_pb,
            inputs=['x:0'],
            outputs=['Identity:0']
        )
        self.input_name = 'images'
        return

    def inference(self: YoloV5TF, sess: Session) -> np.ndarray:
        pred = self.model(tf.convert_to_tensor(
            sess.yolov5_input[self.input_name]
        ))
        pred = tf.squeeze(pred[0]).numpy()
        pred[:, :4] = pred[:, :4] * IMAGE_SIZE
        return pred


class YoloV5OnnxTF(Framework):
    def __init__(self: YoloV5OnnxTF, config: Config) -> None:
        super().__init__(config=config)
        path_weight = f'{path_wt}/onnx_tf_{config.model}'
        if not os.path.isdir(path_weight):
            raise SystemError(f'weight({path_weight}) not found')
        model_sm = tf.keras.models.load_model(path_weight)
        self.model = WrapperYoloV5(yolov5=model_sm)
        self.input_name = 'images'
        return

    def inference(self: YoloV5OnnxTF, sess: Session) -> np.ndarray:
        pred = self.model(sess.yolov5_input[self.input_name])
        return np.squeeze(pred[0].numpy(), 0).copy()


class YoloV5Vino(Framework):
    def __init__(self: YoloV5Vino, config: Config) -> None:
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
        assert input_shape[2] == IMAGE_SIZE
        assert input_shape[3] == IMAGE_SIZE
        output_blob = list(net.outputs.keys())
        assert 'output' in output_blob
        self.output_blob = ['output']
        self.exec_net = ie.load_network(network=net, device_name='CPU')
        return

    def inference(self: YoloV5Vino, sess: Session) -> np.ndarray:
        pred = self.exec_net.infer(inputs=sess.yolov5_input)
        pred = [pred[ob] for ob in self.output_blob]
        return np.squeeze(pred[0], 0).copy()


class YoloV5Onnx(Framework):
    def __init__(self: YoloV5Onnx, config: Config) -> None:
        super().__init__(config=config)
        path_model = f'{path_wt}/{config.model}.onnx'
        if not os.path.isfile(path_model):
            raise SystemError(f'onnx({path_model}) not found')
        self.sess = rt.InferenceSession(path_model)
        input_blob = [x.name for x in self.sess.get_inputs()]
        assert len(input_blob) == 1 and input_blob[0] == 'images'
        self.input_name = input_blob[0]
        input_shape = self.sess.get_inputs()[0].shape
        assert input_shape[2] == IMAGE_SIZE
        assert input_shape[3] == IMAGE_SIZE
        output_blob = [x.name for x in self.sess.get_outputs()]
        assert 'output' in output_blob
        self.output_blob = ['output']
        return

    def inference(self: YoloV5Onnx, sess: Session) -> np.ndarray:
        pred = self.sess.run(
            output_names=self.output_blob,
            input_feed=sess.yolov5_input
        )
        return np.squeeze(pred[0], 0).copy()


class YoloV5Torch(Framework):
    def __init__(self: YoloV5Torch, config: Config) -> None:
        super().__init__(config=config)
        path_torch = f'{path_wt}/{config.model}.pth'
        if not os.path.isfile(path_torch):
            raise SystemError(f'weight({path_torch}) not found')
        repo = 'ultralytics/yolov5'
        model = torch.hub.load(repo, config.model, pretrained=False)
        model.load_state_dict(torch.load(path_torch, map_location='cpu'))
        self.model = model.fuse()
        self.model.eval()
        self.input_name = 'images'
        return

    def inference(self: YoloV5Torch, sess: Session) -> np.ndarray:
        input_feed = torch.from_numpy(
            sess.yolov5_input[self.input_name]
        ).to('cpu')
        with torch.no_grad():
            pred = self.model(input_feed, augment=True)[0]
        return np.squeeze(pred.detach().numpy(), 0).copy()


class YoloV5(Model):
    def __init__(self: YoloV5, config: Config) -> None:
        super().__init__(config=config)
        if config.framework == 'torch':
            self.framework = YoloV5Torch(config=config)
        elif config.framework == 'torch_onnx':
            self.framework = YoloV5Onnx(config=config)
        elif config.framework == 'onnx_vino':
            self.framework = YoloV5Vino(config=config)
        elif config.framework == 'onnx_tf':
            self.framework = YoloV5OnnxTF(config=config)
        elif config.framework == 'tf':
            self.framework = YoloV5TF(config=config)
        elif config.framework == 'tflite':
            self.framework = YoloV5TFLite(config=config)
        elif config.framework == 'tf_onnx':
            self.framework = YoloV5TFOnnx(config=config)
        else:
            raise SystemError(
                f'YOLO V5 unsupport {config.framework}'
            )
        return

    def prep_image(self: YoloV5, sess: Session) -> None:
        sess.padding_image(
            model_height=IMAGE_SIZE, model_width=IMAGE_SIZE
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
        sess.yolov5_input = {self.framework.input_name: image}
        return

    def inference(self: YoloV5, sess: Session) -> np.ndarray:
        pred = super().inference(sess=sess)
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
        # catgory of bouding box is the most plausible category
        cat = cat_conf.argmax(axis=1)[:, np.newaxis].astype(np.float)
        # confidence score of bbox is that of the most plausible category
        conf = cat_conf.max(axis=1)[:, np.newaxis]
        # ready for NMS (0-3: xyxy, 4: category id, 5: confidence score)
        return np.concatenate((xyxy, cat, conf), axis=1)


class DetectorYoloV5(Detector):
    def __init__(self: DetectorYoloV5, config: Config) -> None:
        super().__init__(config=config)
        self.model = YoloV5(config=config)
        return
