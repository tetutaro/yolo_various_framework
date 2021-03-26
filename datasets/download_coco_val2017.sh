#!/usr/bin/env bash
if [ ! -d "val2017" ]; then
    if [ ! -f "val2017.zip" ]; then
        wget http://images.cocodataset.org/zips/val2017.zip
    fi
    unzip val2017.zip
    rm -f val2017.zip
fi
if [ ! -d "annotations" ]; then
    if [ ! -f "annotations_trainval2017.zip" ]; then
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    fi
    unzip annotations_trainval2017.zip
    rm -f annotations_trainval2017.zip
fi
