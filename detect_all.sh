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
datanames=(${dir//\// })
dataname=${datanames[${#datanames[@]}-1]}
models=(
    "yolov3-tiny" "yolov3" "yolov3-spp" "yolov4-tiny" "yolov4" "yolov4-csp" "yolov4x-mish"
)
frames=(
    "tf" "tf_onnx"
)
modelsv5=(
    "yolov5s" "yolov5m" "yolov5l" "yolov5x"
)
framesv5=(
    "torch" "torch_onnx" "onnx_vino" "onnx_tf" "tf" "tf_onnx"
)
quants=(
    "fp32" "fp16"
)
for frame in ${frames[@]} ; do
    for model in ${models[@]} ; do
        rdir="results/${dataname}/${model}_${frame}"
        if [ ! -d ${rdir} ] ; then
            ./detect.py -m ${model} -f ${frame} -d ${dir}
        fi
    done
done
for quant in ${quants[@]} ; do
    for model in ${models[@]} ; do
        rdir="results/${dataname}/${model}_tflite_${quant}"
        if [ ! -d ${rdir} ] ; then
            ./detect.py -m ${model} -f tflite -q ${quant} -d ${dir}
        fi
    done
done
for frame in ${framesv5[@]} ; do
    for model in ${modelsv5[@]} ; do
        rdir="results/${dataname}/${model}_${frame}"
        if [ ! -d ${rdir} ] ; then
            ./detect.py -m ${model} -f ${frame} -d ${dir}
        fi
    done
done
for quant in ${quants[@]} ; do
    for model in ${modelsv5[@]} ; do
        rdir="results/${dataname}/${model}_tflite_${quant}"
        if [ ! -d ${rdir} ] ; then
            ./detect.py -m ${model} -f tflite -q ${quant} -d ${dir}
        fi
    done
done
