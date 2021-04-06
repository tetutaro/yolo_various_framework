## downdload COCO dataset and create small dataset

- download COCO dataset (val2017) and its annotations
    - `> ./download_coco_val2017.sh`
- create small dataset and convert annotations to json lines format
    - `> ./create_small_dataset.py`

## usage

```
usage: create_small_dataset.py [-h] [--number NUMBER] [--directory DIRECTORY]

create small dataset from COCO val2017 dataset

optional arguments:
  -h, --help            show this help message and exit
  --number NUMBER, -n NUMBER
                        number of images (default: 10)
  --directory DIRECTORY, -d DIRECTORY
                        directory name (defalt: "sample_dataset")
```
