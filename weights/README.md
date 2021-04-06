# download YOLO pre-trained weights

## YOLO V3 and V4

`> ./download_yolo.py`

## YOLO V5

`> ./download_yolov5.py`

## [optional] compile TFLite Flat Buffers for EdgeTPU

### setup

- install docker on your PC
- `> build_docker.sh`
- convert pre-trained weights
    - go to the directory up
    - `> convert_yolo.py`
    - `> convert_yolov5.py`

### compile

- run docker
- ex.) compile yolov3-tiny for Edge TPU
    - `> compile_edgetpu.sh yolo/yolov3-tiny_int8.tflite`
    - → `yolo/yolov3-tiny_int8_edgetpu.tflite` will be created
- ex.) compile yolov5s for Edge TPU
    - `> compile_edgetpu.sh yolov5/yolov5s_int8.tflite`
    - → `yolov5/yolov5s_int8_edgetpu.tflite` will be created

### notices

- quantization of compiled model must be `int8`
    - the filename is `[yolo|yolov5]/*_int8.tflite`
- It is toooo slow using compiled binery because most of subgraph are not mapped on TPU.

the case of yolov3-tiny
```
> ./compile_edgetpu.sh yolo/yolov3-tiny_int8.tflite
Edge TPU Compiler version 15.0.340273435

Model compiled successfully in 577 ms.

Input model: /home/yolo/yolov3-tiny_int8.tflite
Input size: 8.58MiB
Output model: /home/yolo/yolov3-tiny_int8_edgetpu.tflite
Output size: 8.70MiB
On-chip memory used for caching model parameters: 3.00KiB
On-chip memory remaining for caching model parameters: 7.67MiB
Off-chip memory used for streaming uncached model parameters: 0.00B
Number of Edge TPU subgraphs: 1
Total number of operations: 36
Operation log: /home/yolo/yolov3-tiny_int8_edgetpu.log

Model successfully compiled but not all operations are supported by the Edge TPU. A percentage of the model will instead run on the CPU, which is slower. If possible, consider updating your model to use only operations supported by the Edge TPU. For details, visit g.co/coral/model-reqs.
Number of operations that will run on Edge TPU: 2
Number of operations that will run on CPU: 34

Operator                       Count      Status

MAX_POOL_2D                    6          More than one subgraph is not supported
QUANTIZE                       2          Operation is otherwise supported, but not mapped due to some unspecified limitation
QUANTIZE                       1          Mapped to Edge TPU
QUANTIZE                       1          More than one subgraph is not supported
CONV_2D                        1          Mapped to Edge TPU
CONV_2D                        12         More than one subgraph is not supported
RESIZE_NEAREST_NEIGHBOR        1          Operation version not supported
LEAKY_RELU                     11         Operation not supported
CONCATENATION                  1          More than one subgraph is not supported
```

the case of yolov5s
```
> ./compile_edgetpu.sh yolov5/yolov5s_int8.tflite
Edge TPU Compiler version 15.0.340273435

Model compiled successfully in 93 ms.

Input model: /home/yolov5/yolov5s_int8.tflite
Input size: 7.39MiB
Output model: /home/yolov5/yolov5s_int8_edgetpu.tflite
Output size: 7.34MiB
On-chip memory used for caching model parameters: 0.00B
On-chip memory remaining for caching model parameters: 8.05MiB
Off-chip memory used for streaming uncached model parameters: 0.00B
Number of Edge TPU subgraphs: 1
Total number of operations: 294
Operation log: /home/yolov5/yolov5s_int8_edgetpu.log

Model successfully compiled but not all operations are supported by the Edge TPU. A percentage of the model will instead run on the CPU, which is slower. If possible, consider updating your model to use only operations supported by the Edge TPU. For details, visit g.co/coral/model-reqs.
Number of operations that will run on Edge TPU: 1
Number of operations that will run on CPU: 293

Operator                       Count      Status

QUANTIZE                       1          Mapped to Edge TPU
QUANTIZE                       1          Operation is otherwise supported, but not mapped due to some unspecified limitation
QUANTIZE                       24         More than one subgraph is not supported
LOGISTIC                       62         More than one subgraph is not supported
TRANSPOSE                      3          Operation not supported
SUB                            3          More than one subgraph is not supported
CONCATENATION                  1          Operation is otherwise supported, but not mapped due to some unspecified limitation
CONCATENATION                  17         More than one subgraph is not supported
MAX_POOL_2D                    3          More than one subgraph is not supported
STRIDED_SLICE                  9          More than one subgraph is not supported
STRIDED_SLICE                  4          Only Strided-Slice with unitary strides supported
CONV_2D                        62         More than one subgraph is not supported
MUL                            80         More than one subgraph is not supported
RESIZE_NEAREST_NEIGHBOR        2          Operation version not supported
RESHAPE                        3          Operation is otherwise supported, but not mapped due to some unspecified limitation
RESHAPE                        3          More than one subgraph is not supported
PAD                            6          More than one subgraph is not supported
ADD                            10         More than one subgraph is not supported
```
