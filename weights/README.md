# download and convert YOLO pre-trained weights

## YOLO V3 and V4

`> ./download_yolo.py`

## YOLO V5

`> ./download_yolov5.py`

## [optional] compile TFLite Flat Buffers for EdgeTPU

- install docker on your PC
- run docker
- `> build_docker.sh`
- `> download_coco.sh`
- uncomment commented source codes in `convert_tf_tflite.py` and re-run
- `> compile_edgetpu.sh yolov5[s|m|l|x]_int8.tflite`
