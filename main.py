#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Dropout, BatchNormalization, Reshape
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

def main():
    model = Sequential()
    model.add(InputLayer((128, 128, 2)))

    # 4 Layers with 64 filters, then another 4 with 128 filters
    filters = 4 * [64] + 4 * [128]
    for i, f in enumerate(filters, 1):
        model.add(Conv2D(f, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        # MaxPooling after every 2 Conv layers
        if i % 2 == 0:
            model.add(MaxPooling2D(strides=(2, 2)))

    # Flatten
    model.add(Reshape((1, -1)))

    model.add(Dense(1024))
    model.add(Dropout(0.5))

    # Regression model
    model.add(Dense(8))

    # "Decrease the learning rate by a factor of 10 after every 30,000 iterations"
    decay = (1.0/10)/30000
    sgd = SGD(lr=0.005, decay=decay, momentum=0.9, nesterov=True)

    # For a mean squared error regression problem
    model.compile(optimizer=sgd,
                  loss='mean_squared_error')

    plot_model(model, show_shapes=True)


if __name__ == '__main__':
    main()
