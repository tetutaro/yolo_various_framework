#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import Generator, Dict
import os
import time
import glob
import subprocess
import simplejson as json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
from utils.nms import filter_bboxes

ld_library_path = subprocess.Popen(
    'pyenv which python', stdout=subprocess.PIPE, shell=True
).communicate()[0].decode('utf-8').strip().split(os.sep)[:-2]
ld_library_path.append('lib')
ld_library_path = os.sep.join(ld_library_path)
os.environ['LD_LIBRARY_PATH'] = ld_library_path


class Session(object):
    def __init__(self: Session, path: str) -> None:
        self.name = os.path.basename(path)
        self.image_id = os.path.splitext(self.name)[0]
        self.raw_image = cv2.imread(path)
        self.raw_height = self.raw_image.shape[0]
        self.raw_width = self.raw_image.shape[1]
        self.fine_image = self.raw_image.copy()
        self.model_height = None
        self.model_width = None
        self.scale = None
        self.image_height = None
        self.image_width = None
        self.offset_height = None
        self.offset_width = None
        self.pad_image = None
        self.yolo_input = None
        self.yolov5_input = None
        self.crowddet_input = None
        self.elapsed_ms = None
        self.pred_count = None
        self.pred_bboxes = None
        return

    def padding_image(
        self: Session,
        model_height: int,
        model_width: int
    ) -> None:
        # calc height, width, offset to resize
        self.model_height = model_height
        self.model_width = model_width
        self.scale = min(
            float(self.model_height) / self.raw_height,
            float(self.model_width) / self.raw_width
        )
        self.image_height = int(round(self.raw_height * self.scale))
        self.image_width = int(round(self.raw_width * self.scale))
        self.offset_height = (
            self.model_height - self.image_height
        ) // 2
        self.offset_width = (
            self.model_width - self.image_width
        ) // 2
        # resize image and put it on to the background image
        if self.scale >= 1:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
        img = cv2.resize(
            self.fine_image.copy(),
            (self.image_width, self.image_height),
            interpolation=interpolation
        )
        background = np.full(
            (self.model_height, self.model_width, 3),
            128, dtype=np.uint8
        )
        background[
            self.offset_height: self.offset_height + self.image_height,
            self.offset_width: self.offset_width + self.image_width,
        ] = img
        self.pad_image = background
        return

    def rescale_xyxy(self: Session, xyxy: np.ndarray) -> np.ndarray:
        xyxy[:, 0] = np.maximum(
            (xyxy[:, 0] - self.offset_width) / self.scale,
            0
        )
        xyxy[:, 1] = np.maximum(
            (xyxy[:, 1] - self.offset_height) / self.scale,
            0
        )
        xyxy[:, 2] = np.minimum(
            (xyxy[:, 2] - self.offset_width) / self.scale,
            self.raw_width
        )
        xyxy[:, 3] = np.minimum(
            (xyxy[:, 3] - self.offset_height) / self.scale,
            self.raw_height
        )
        return xyxy

    def draw_prediction(self: Session, category_map: Dict) -> Image:
        fontsize = int(round(max(
            self.raw_width, self.raw_height
        ) / 32.0))
        buf = self.raw_image.copy()
        buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        buf = Image.fromarray(buf)
        draw = ImageDraw.Draw(buf)
        font = ImageFont.truetype(
            font='TakaoGothic.ttf',
            size=fontsize
        )
        default_color = (0xff, 0xff, 0xff)
        margin = 5
        bboxes = self.pred_bboxes.tolist()
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, cls, prob = bbox
            xmin = int(max(xmin, margin))
            ymin = int(max(ymin, margin))
            xmax = int(min(xmax, self.raw_width - margin))
            ymax = int(min(ymax, self.raw_height - margin))
            cls = int(cls)
            prob = round(prob, 3)
            name = category_map[cls]['category_name']
            if name is None:
                name = str(cls)
            color = tuple(np.array(np.array(cm.jet((
                prob - 0.25  # self.config.confidence_threshold
            ) / (
                1.0 - 0.25  # self.config.confidence_threshold
            ))) * 255, dtype=np.uint8).tolist())
            color = tuple(color[0:3])
            draw.rectangle(
                (xmin, ymin, xmax, ymax),
                fill=None, outline=color, width=2
            )
            draw.text(
                (xmin + 5, ymin + 5), name,
                fill=color, font=font
            )
            draw.text(
                (xmin + 5, ymin + 5 + fontsize), '%0.3f' % prob,
                fill=color, font=font
            )
        text = 'Elapsed Time: %d[ms]' % self.elapsed_ms
        draw.text((5, 5), text, fill=default_color, font=font)
        text = 'Detected Objects: %d' % self.pred_count
        draw.text((5, 5 + fontsize), text, fill=default_color, font=font)
        return buf


class Config(object):
    def __init__(
        self: Config,
        model: str,
        framework: str,
        quantize: str,
        image_dir: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.model = model
        self.framework = framework
        self.quantize = quantize
        self.image_dir = image_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        return


class Framework(object):
    def __init__(
        self: Framework,
        config: Config
    ) -> None:
        self.config = config
        self.input_blob = list()
        self.input_shape = [None, None, None, None]
        return

    def inference(self: Framework, sess: Session) -> np.ndarray:
        return np.empty((0, 6), dtype=float)


class Model(object):
    def __init__(
        self: Model,
        config: Config
    ) -> None:
        self.config = config
        # self.framework= Framework(config=config)
        self.framework = None
        self.category_map = self.read_labels()
        return

    def read_labels(self: Model) -> Dict:
        category_map = dict()
        with open('labels/coco_labels.txt', 'rt') as rf:
            for i, raw in enumerate(rf.read().strip().splitlines()):
                id_label = raw.strip().split(' ', 1)
                category_id = int(id_label[0])
                category_name = id_label[1].strip()
                category_map[i] = {
                    'coco_category_id': category_id + 1,
                    'category_name': category_name,
                }
        return category_map

    def prep_image(self: Model, sess: Session) -> None:
        return

    def inference(self: Model, sess: Session) -> np.ndarray:
        return self.framework.inference(sess=sess)


class Detector(object):
    def __init__(
        self: Detector,
        config: Config
    ) -> None:
        self.config = config
        # self.model = Model(config=config)
        self.model = None
        dataset_name = config.image_dir.split(os.sep)[-1]
        if self.config.framework == 'tflite':
            self.result_dir = 'results/%s/%s_%s_%s' % (
                dataset_name, config.model,
                config.framework, config.quantize
            )
        else:
            self.result_dir = 'results/%s/%s_%s' % (
                dataset_name, config.model, config.framework
            )
        os.makedirs(self.result_dir, exist_ok=True)
        self.wf = open(os.path.join(
            self.result_dir, 'predictions.jsonl'
        ), 'wt')
        return

    def print_header(self: Detector) -> None:
        if self.config.framework == 'tflite':
            framework = f'{self.config.framework}_{self.config.quantize}'
        else:
            framework = self.config.framework
        print(
            '=== MODEL: %s, FRAMEWORK: %s ===' % (
                self.config.model, framework
            )
        )
        return

    def yield_session(self: Detector) -> Generator[Session, None, None]:
        for path in sorted(glob.glob(f'{self.config.image_dir}/*')):
            if not path.endswith(('.jpg', '.png')):
                continue
            yield Session(path=path)
        return

    def prep_image(self: Detector, sess: Session) -> None:
        self.model.prep_image(sess=sess)
        return

    def inference(self: Detector, sess: Session) -> None:
        # image preprocessing
        self.prep_image(sess=sess)
        # inference
        start_time = time.perf_counter()
        pred = self.model.inference(sess=sess)
        pred = filter_bboxes(
            pred,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            is_soft=True
        )
        end_time = time.perf_counter()
        # sort bounding boxes by confidence ascending
        pred = pred[np.argsort(pred[:, 5])]
        # results of this session
        sess.elapsed_ms = int(round((end_time - start_time) * 1000))
        sess.pred_count = pred.shape[0]
        sess.pred_bboxes = pred
        return

    def print_result(self: Detector, sess: Session) -> None:
        print(
            '%s: count=%d, time=%dms' % (
                sess.name, sess.pred_count, sess.elapsed_ms
            )
        )
        return

    def dump_result(self: Detector, sess: Session) -> None:
        if self.config.framework == 'tflite':
            framework = f'{self.config.framework}_{self.config.quantize}'
        else:
            framework = self.config.framework
        bboxes = list()
        for pbox in sess.pred_bboxes.tolist():
            box = [
                float(pbox[0]), float(pbox[1]),
                float(pbox[2] - pbox[0]), float(pbox[3] - pbox[1])
            ]
            bbox = {
                'category_id': self.model.category_map[
                    int(pbox[4])
                ]['coco_category_id'],
                'bbox': box,
                'score': float(pbox[5]),
            }
            bboxes.append(bbox)
        stats = {
            'model': self.config.model,
            'framework': framework,
            'filename': sess.name,
            'count': sess.pred_count,
            'msec': sess.elapsed_ms,
            'image_id': sess.image_id,
            'bboxes': bboxes,
        }
        self.wf.write(json.dumps(stats) + '\n')
        self.wf.flush()
        return

    def dump_image(self: Detector, sess: Session) -> None:
        image = sess.draw_prediction(category_map=self.model.category_map)
        path_image = os.path.join(self.result_dir, sess.name)
        image.save(path_image)
        return

    def close(self: Detector) -> None:
        self.wf.close()
        return
