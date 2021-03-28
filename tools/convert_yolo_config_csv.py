#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import OrderedDict
import configparser
import os
import pandas as pd
import argparse


class LayerDict(OrderedDict):
    def __init__(self):
        super().__init__()
        self.seq = 0

    def __setitem__(self, key, val):
        if key.startswith('#'):
            return
        if isinstance(val, dict):
            if key == 'net':
                return
            self.seq += 1
            key = f'{self.seq}:{key}'
        super().__setitem__(key, val)
        return


def main(config: str) -> None:
    if not os.path.isfile(config):
        raise ValueError(f'config({config}) is not exists')
    cfg = configparser.ConfigParser(
        defaults=None,
        dict_type=LayerDict,
        strict=False,
        empty_lines_in_values=False,
        comment_prefixes=(';', '#'),
        allow_no_value=True
    )
    cfg.read(config)
    layers = list()
    convs = 0
    for i, section in enumerate(cfg.sections()):
        layer_type = section.split(':')[1]
        layer = dict(cfg.items(section))
        layer['no.'] = i
        layer['type'] = layer_type
        if layer_type == 'convolutional':
            layer['convs'] = convs
            convs += 1
        layers.append(layer)
    df = pd.DataFrame(layers)
    needed_columns = [
        'type', 'convs', 'filters', 'size', 'stride', 'pad',
        'activation', 'batch_normalize',
        'no.', 'from', 'layers'
    ]
    for nc in needed_columns:
        if nc not in list(df.columns):
            df[nc] = [None] * df.shape[0]
    df = df[needed_columns]
    csv = config.replace('.cfg', '.csv')
    df.to_csv(csv, encoding='utf_8_sig', index=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert YOLO config file(.cfg) to csv (utf-8 BOM)'
    )
    parser.add_argument(
        'config', type=str, help='YOLO config file'
    )
    args = parser.parse_args()
    main(**vars(args))
