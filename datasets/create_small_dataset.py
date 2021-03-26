#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Union
import os
import glob
import shutil
from collections import defaultdict
import numpy as np
import simplejson as json
import argparse


def convert_xywh_xyxy(
    xywh: List[Union[int, float]]
) -> List[float]:
    return [
        float(xywh[0]),
        float(xywh[1]),
        float(xywh[0] + xywh[2]),
        float(xywh[1] + xywh[3])
    ]


def create_dataset(number: int, directory: str) -> None:
    # delete old dataset and create new dataset
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    # select images randomly
    fns = glob.glob('val2017/*')
    fns = [
        os.path.basename(x) for x in fns
        if x.endswith(('.png', '.jpg'))
    ]
    np.random.shuffle(fns)
    fns = sorted(fns[:number])
    # convert COCO annotations to the format of `object_detection_metrics`
    with open('annotations/instances_val2017.json', 'rt') as rf:
        orig_anns = json.load(fp=rf)
    bboxes = defaultdict(list)
    for ann in orig_anns['annotations']:
        image_id = '%012d' % ann['image_id']
        bboxes[image_id].append({
            'category_id': ann['category_id'],
            'bbox': convert_xywh_xyxy(ann['bbox']),
        })
    # copy images and dump annotations
    new_anns = list()
    for fn in fns:
        shutil.copy(f'val2017/{fn}', f'{directory}/{fn}')
        image_id = os.path.splitext(fn)[0]
        new_anns.append({
            'image_id': image_id,
            'bboxes': bboxes[image_id]
        })
    with open(f'{directory}/ground_truths.jsonl', 'wt') as wf:
        for ann in new_anns:
            wf.write(json.dumps(ann) + '\n')
    return


if __name__ == '__main__':
    if not (os.path.isdir('val2017') and os.path.isdir('annotations')):
        raise SystemError('run `download_coco_val2017.sh` first')
    parser = argparse.ArgumentParser(
        description='create small dataset from COCO val2017 dataset'
    )
    parser.add_argument(
        '--number', '-n', type=int, default=10,
        help='number of images (default: 10)'
    )
    parser.add_argument(
        '--directory', '-d', type=str, default='sample_dataset',
        help='directory name (defalt: "sample_dataset")'
    )
    args = parser.parse_args()
    create_dataset(**vars(args))
