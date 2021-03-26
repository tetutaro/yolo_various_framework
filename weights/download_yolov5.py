#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from pathlib import Path
import requests
import torch

directory = 'yolov5'


def download_weight(
    model: str,
    repo: str = 'ultralytics/yolov5'
) -> None:
    assets = [
        'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'
    ]
    assert model in assets, 'invalid model name'
    fpath = Path(os.path.join(
        directory, model.strip().replace("'", '').lower()
    ))
    if fpath.exists():
        return
    try:
        response = requests.get(
            f'https://api.github.com/repos/{repo}/releases/latest'
        ).json()  # github api
        tag = response['tag_name']
    except Exception:
        print('cannot get tag name')
        return
    name = fpath.name
    redundant = False  # second download option
    try:  # GitHub
        url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
        print(f'Downloading {model}...')
        torch.hub.download_url_to_file(url, fpath)
        # check
        assert fpath.exists() and fpath.stat().st_size > 1E6
    except Exception as e:  # GCP
        print(f'Download error: {e}')
        assert redundant, 'No secondary mirror'
        url = f'https://storage.googleapis.com/{repo}/ckpt/{name}'
        print(f'Downloading {model}...')
        os.system(f'curl -L {url} -o {name}')
    finally:
        # check
        if not fpath.exists() or fpath.stat().st_size < 1E6:
            # remove partial downloads
            fpath.unlink(missing_ok=True)
            print('ERROR: Download failure')
    return


if __name__ == '__main__':
    os.makedirs(directory, exist_ok=True)
    for x in ['s', 'm', 'l', 'x']:
        if os.path.isfile(f'{directory}/yolov5{x}.pt'):
            continue
        download_weight(model=f'yolov5{x}.pt')
