#!/bin/sh
if [ $# != 1 ]; then
    echo "Usage: $0 <model.tflite>"
    exit
elif [ ! -f $1 ] || [ ${1##*.} != 'tflite' ]; then
    echo "Usage: $0 <model.tflite>"
    exit
fi
docker run --rm -it --env 'TZ=Asia/Tokyo' --volume ${PWD}:/home --name edgetpu_compile edgetpu_env /usr/bin/edgetpu_compiler -s /home/$1 -o /home
