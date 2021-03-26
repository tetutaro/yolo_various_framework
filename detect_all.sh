#!/usr/bin/env bash
if [ $# != 1 ]; then
    echo "Usage: $0 [dir]"
    exit 1
fi
dir=$1
if [ ! -d ${dir} ]; then
    echo "${dir} not found"
    exit 1
fi
for frame in torch torch_onnx onnx_vino onnx_tf tf tf_onnx ; do
    for model in yolov5s yolov5m yolov5l yolov5x ; do
        ./detect.py -m ${model} -f ${frame} -d ${dir}
    done
done
for quant in fp32 fp16 ; do
    for model in yolov5s yolov5m yolov5l yolov5x ; do
        ./detect.py -m ${model} -f tflite -q ${quant} -d ${dir}
    done
done
