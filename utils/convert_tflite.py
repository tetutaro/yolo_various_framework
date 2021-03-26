#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List
import os
import time
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2
)

NUM_TRAINING_IMAGES = 100


def save_frozen_graph(
    path_pb: str,
    model_keras: tf.keras.Model
) -> None:
    if os.path.isfile(path_pb):
        return
    full_model = tf.function(lambda x: model_keras(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(
            model_keras.inputs[0].shape,
            model_keras.inputs[0].dtype
        )
    )
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    # check inputs and outputs of frozen graph
    # print(frozen_func.inputs)
    # print(frozen_func.outputs)
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=os.path.dirname(path_pb),
        name=os.path.basename(path_pb),
        as_text=False
    )
    return


def load_frozen_graph(path_pb: str) -> tf.function:
    with tf.io.gfile.GFile(path_pb, "rb") as rf:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(rf.read())

    def wrap_frozen_graph(graph_def, inputs, outputs):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(
            _imports_graph_def, []
        )
        import_graph = wrapped_import.graph
        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs)
        )

    # the name of inputs and outputs can be known with printing
    # frozen_func.inputs/outputs when `save_frozen_graph()`
    frozen_func = wrap_frozen_graph(
        graph_def=graph_def,
        inputs=['x:0'],
        outputs=['Identity:0']
    )
    return frozen_func


def convert_tflite_fp32(
    path_tflite: str,
    model_keras: tf.keras.Model
) -> None:
    if os.path.isfile(path_tflite):
        return
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.allow_custom_ops = False
    converter.experimental_new_converter = True
    model_tflite = converter.convert()
    open(path_tflite, "wb").write(model_tflite)
    return


def convert_tflite_fp16(
    path_tflite: str,
    model_keras: tf.keras.Model
) -> None:
    if os.path.isfile(path_tflite):
        return
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]
    converter.target_spec.supported_types = [
        tf.float16
    ]
    converter.allow_custom_ops = False
    converter.experimental_new_converter = True
    model_tflite = converter.convert()
    open(path_tflite, "wb").write(model_tflite)
    return


def convert_tflite_int8(
    path_tflite: str,
    imgsize: List[int],
    model_keras: tf.keras.Model
) -> None:
    if not os.path.isdir('datasets/val2017'):
        raise SystemError(
            'you need COCO 2017 val dataset for post-training'
        )
    if os.path.isfile(path_tflite):
        return

    def representative_dataset_gen():
        images = glob.glob('datasets/val2017/*.jpg')
        np.random.shuffle(images)
        for i, ipath in enumerate(images[:NUM_TRAINING_IMAGES]):
            img = cv2.imread(ipath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ih = img.shape[0]
            iw = img.shape[1]
            scale = min(imgsize[0] / ih, imgsize[1] / iw)
            nh = int(ih * scale)
            nw = int(iw * scale)
            oh = (imgsize[0] - nh) // 2
            ow = (imgsize[1] - nw) // 2
            if scale >= 1:
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_AREA
            nimg = cv2.resize(
                img.copy(), (nw, nh),
                interpolation=interpolation
            )
            rimg = np.full((*imgsize, 3), 128, dtype=np.uint8)
            rimg[oh:oh + nh, ow:ow + nw, :] = nimg
            rimg = rimg[np.newaxis, ...].astype(np.float32)
            rimg /= 255.0
            yield [rimg]
            if i % 10 == 9:
                print(f'post-training... ({i}/{NUM_TRAINING_IMAGES})')
        return

    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    model_tflite = converter.convert()
    open(path_tflite, "wb").write(model_tflite)
    return


def _print_detail(details: List) -> None:
    for i, detail in enumerate(details):
        print("{}: index={} shape={} dtype={}".format(
            i, detail['index'], detail['shape'], detail['dtype']
        ))
    return


def test_tflite(path_tflite: str, mode: str) -> None:
    assert mode in ['fp32', 'fp16', 'int8']
    if not os.path.isfile(path_tflite):
        print(f'ERROR: {path_tflite} not found')
        return
    print(f'MODEL: {path_tflite}')
    interpreter = tf.lite.Interpreter(path_tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print('input details')
    _print_detail(input_details)
    output_details = interpreter.get_output_details()
    print('output details')
    _print_detail(output_details)
    input_shape = input_details[0]['shape']
    input_data = np.array(
        np.random.randint(0, 256, input_shape)
    )
    if mode == 'int8':
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (input_data / 255.0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()
    elapsed = round((end_time - start_time) * 1000, 3)
    print(f'elapsed time taken for inference: {elapsed}[ms]')
    output_data = [
        interpreter.get_tensor(
            output_details[i]['index']
        ) for i in range(len(output_details))
    ]
    for i, out in enumerate(output_data):
        out_shape = output_details[i]['shape']
        assert len(out.shape) == len(out_shape)
        for j, v in enumerate(out.shape):
            assert v == out_shape[j]
    return
