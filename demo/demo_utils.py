#!/usr/bin/env python

import cv2
import numpy as np

from dataset.generate import *


def mean_corner_error(y_true, y_pred):
    y_true = np.reshape(y_true, (-1, 4, 2))
    y_pred = np.reshape(y_pred, (-1, 4, 2))
    return np.mean(np.sqrt(np.sum(np.square(y_pred - y_true), axis=-1, keepdims=True)), axis=1)


def draw_lines(img, points, color, thickness=2):
    out = img.copy()
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        cv2.line(out, tuple(points[i]), tuple(points[j]), color, thickness, cv2.LINE_AA)
    return out


def transform_points(p_original, p_perturbed):
    # See: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
    M = cv2.getPerspectiveTransform(np.float32(p_perturbed), np.float32(p_original))
    # Convert to homogenous representation of points
    h_points = np.hstack((p_original, np.array([1, 1, 1, 1])[np.newaxis].T))
    # Transform
    h_points = M.dot(h_points.T)
    t = h_points[2]
    h_points = h_points[:2] / t
    return h_points.T.astype('uint16')
