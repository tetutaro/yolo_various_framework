#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np


# Intersection of Union
# bboxesX[:4] is numpy array of xyxy (xmin, ymin, xmax, ymax)
# bboxes1: the bounding box which has the highest confidence score
# bboxes2: the bounding boxes of same category expect above
def bboxes_iou(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    bboxes1_area = (
        bboxes1[:, 2] - bboxes1[:, 0]
    ) * (
        bboxes1[:, 3] - bboxes1[:, 1]
    )
    bboxes2_area = (
        bboxes2[:, 2] - bboxes2[:, 0]
    ) * (
        bboxes2[:, 3] - bboxes2[:, 1]
    )
    left_ups = np.maximum(bboxes1[:, :2], bboxes2[:, :2])
    right_downs = np.minimum(bboxes1[:, 2:4], bboxes2[:, 2:4])
    intersections = np.maximum(right_downs - left_ups, 0.0)
    inter_areas = intersections[:, 0] * intersections[:, 1]
    union_areas = bboxes1_area + bboxes2_area - inter_areas
    ious = np.maximum(
        1.0 * inter_areas / union_areas,
        np.finfo(np.float32).eps
    )
    # if the bouding box of bboxes2 is a subset of bboxes1,
    # set IoU as 1.0 (should be removed)
    is_subset = (
        bboxes1[:, 0] <= bboxes2[:, 0]
    ) * (
        bboxes1[:, 1] <= bboxes2[:, 1]
    ) * (
        bboxes1[:, 2] >= bboxes2[:, 2]
    ) * (
        bboxes1[:, 3] >= bboxes2[:, 3]
    )
    ious = np.maximum(ious, is_subset)
    return ious


# filter bounding boxes using (soft) Non-Maximum Suppression
# paper of soft NMS: https://arxiv.org/abs/1704.04503
# bboxes is numpy array of
# offset 0-3: xyxy (xmin, ymin, xmax, ymax)
# offset 4: category id (int)
# offset 5: confidence score
def filter_bboxes(
    bboxes: np.ndarray,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.45,
    is_soft: bool = True
) -> np.ndarray:
    if bboxes.shape[0] == 0:
        return bboxes
    # filter by confidence threshold
    bboxes = bboxes[bboxes[:, 5] > conf_threshold]
    if bboxes.shape[0] == 0:
        return bboxes
    # confidence for soft NMS
    bboxes = np.insert(bboxes, 6, bboxes[:, 5], axis=1)
    # (soft) NMS for each class
    unique_category_ids = list(set(bboxes[:, 4]))
    best_bboxes = list()
    for cat in unique_category_ids:
        cat_bboxes = bboxes[bboxes[:, 4] == cat]
        while cat_bboxes.shape[0] > 0:
            if cat_bboxes.shape[0] == 1:
                best_bboxes.append(cat_bboxes)
                break
            max_conf = np.argmax(cat_bboxes[:, 6])
            best_bbox = cat_bboxes[max_conf:max_conf + 1]
            best_bboxes.append(best_bbox)
            cat_bboxes = np.delete(cat_bboxes, max_conf, axis=0)
            ious = bboxes_iou(best_bbox, cat_bboxes)
            if is_soft:
                iou_mask = (ious >= iou_threshold).astype(np.float)
                cat_bboxes[:, 6] = cat_bboxes[:, 6] * (
                    1.0 - (ious * iou_mask)
                )
                cat_bboxes = cat_bboxes[cat_bboxes[:, 6] > conf_threshold]
            else:
                cat_bboxes = cat_bboxes[ious < iou_threshold]
    return np.concatenate(best_bboxes, axis=0)[:, :6]
