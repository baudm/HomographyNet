#!/usr/bin/env python

from keras import backend as K


def mean_corner_error(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, 4, 2))
    y_pred = K.reshape(y_pred, (-1, 4, 2))
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True)), axis=1)
