#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
from cv2 import dnn_superres_DnnSuperResImpl


def adjust_white_balance(image: np.ndarray) -> np.ndarray:
    # white balance adjustment for strong neutral white
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(image[:, :, 1])
    avg_b = np.average(image[:, :, 2])
    image[:, :, 1] = image[:, :, 1] - (
        (avg_a - 128) * (image[:, :, 0] / 255.0) * 1.1
    )
    image[:, :, 2] = image[:, :, 2] - (
        (avg_b - 128) * (image[:, :, 0] / 255.0) * 1.1
    )
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return image


def smooth_image(image: np.ndarray) -> np.ndarray:
    # image smoothing for noise removal
    return cv2.GaussianBlur(image, (5, 5), 0)


def correct_contrast(image: np.ndarray) -> np.ndarray:
    # contrast correction that brightens dark areas
    b = 10  # HEURISTIC !!
    gamma = 1 / np.sqrt(image.mean()) * b
    g_table = np.array([
        ((i / 255.0) ** (1 / gamma)) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(image, g_table)


def levelize_histogram(image: np.ndarray) -> np.ndarray:
    # make color distributions even
    for i in range(3):
        image[:, :, i] = cv2.equalizeHist(image[:, :, i])
    return image


def correct_contrast_using_lut(image: np.ndarray) -> np.ndarray:
    # contrast correction using look-up-table
    a = 10
    c_table = np.array([
        255.0 / (1 + np.exp(-a * (i - 128) / 255))
        for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(image, c_table)


def upsample_image(
    image: np.ndarray,
    sr: dnn_superres_DnnSuperResImpl
) -> np.ndarray:
    # increase resolution with super-resolution to make the image clearer
    # and then shrink the image
    prev_height = image.shape[0]
    prev_width = image.shape[1]
    # sr.upsample() (super-resolution) is too slow when the image is big
    # so, do super-resolution after shrinking image
    image = cv2.resize(
        image, (prev_width // 2, prev_height // 2),
        interpolation=cv2.INTER_AREA
    )
    image = sr.upsample(image)
    # restore image size
    image = cv2.resize(
        image, (prev_width, prev_height),
        interpolation=cv2.INTER_AREA
    )
    return image
