#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Dropout, \
    BatchNormalization, Flatten


def create_model():
    model = Sequential()
    model.add(InputLayer((128, 128, 2)))

    # 4 Layers with 64 filters, then another 4 with 128 filters
    filters = 4 * [64] + 4 * [128]
    for i, f in enumerate(filters, 1):
        model.add(Conv2D(f, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        # MaxPooling after every 2 Conv layers except the last one
        if i % 2 == 0 and i != 8:
            model.add(MaxPooling2D(strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dropout(0.5))

    # Regression model
    model.add(Dense(8))
    return model
