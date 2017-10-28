#!/usr/bin/env python

import os.path

from keras.applications import MobileNet
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Dropout, \
    BatchNormalization, Flatten, Concatenate


BASELINE_WEIGHTS_PATH = 'https://github.com/baudm/HomographyNet/raw/master/models/homographynet_weights_tf_dim_ordering_tf_kernels.h5'
MOBILENET_WEIGHTS_PATH = 'https://github.com/baudm/HomographyNet/raw/master/models/mobile_homographynet_weights_tf_dim_ordering_tf_kernels.h5'


def create_model(use_weights=False):
    model = Sequential(name='homographynet')
    model.add(InputLayer((128, 128, 2), name='input_1'))

    # 4 Layers with 64 filters, then another 4 with 128 filters
    filters = 4 * [64] + 4 * [128]
    for i, f in enumerate(filters, 1):
        model.add(Conv2D(f, 3, padding='same', activation='relu', name='conv2d_{}'.format(i)))
        model.add(BatchNormalization(name='batch_normalization_{}'.format(i)))
        # MaxPooling after every 2 Conv layers except the last one
        if i % 2 == 0 and i != 8:
            model.add(MaxPooling2D(strides=(2, 2), name='max_pooling2d_{}'.format(int(i/2))))

    model.add(Flatten(name='flatten_1'))
    model.add(Dropout(0.5, name='dropout_1'))
    model.add(Dense(1024, activation='relu', name='dense_1'))
    model.add(Dropout(0.5, name='dropout_2'))

    # Regression model
    model.add(Dense(8, name='dense_2'))

    if use_weights:
        weights_name = os.path.basename(BASELINE_WEIGHTS_PATH)
        weights_path = get_file(weights_name, BASELINE_WEIGHTS_PATH,
                                cache_subdir='models',
                                file_hash='915d92726132f3e1d38b69b64838ef2b5d8bbe8ea223b06c792aa72cce6030a6')
        model.load_weights(weights_path)

    return model


def create_mobilenet_model(use_weights=False):
    base_model = MobileNet(input_shape=(128, 128, 2), include_top=False, weights=None)
    # The output shape just before the pooling and dense layers is: (4, 4, 1024)
    x = base_model.output

    # 4 Conv layers in parallel with 2 4x4 filters each
    x = [Conv2D(2, 4, name='conv2d_{}'.format(i))(x) for i in range(1, 5)]
    x = Concatenate(name='concatenate_1')(x)
    x = Flatten(name='flatten_1')(x)

    model = Model(base_model.input, x, name='mobile_homographynet')

    if use_weights:
        weights_name = os.path.basename(MOBILENET_WEIGHTS_PATH)
        weights_path = get_file(weights_name, MOBILENET_WEIGHTS_PATH,
                                cache_subdir='models',
                                file_hash='e161aabc5a04ff715a6f5706855a339d598d1216a4a5f45b90b8dbf5f8bcedc3')
        model.load_weights(weights_path)

    return model
