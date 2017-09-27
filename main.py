#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Dropout, BatchNormalization, Flatten
from keras.optimizers import SGD as _SGD
from keras.utils.vis_utils import plot_model


class SGD(_SGD):
    """
    SGD optimizer with custom LR decay logic

    To use properly, make sure that decay is always set to 0.
    """

    def get_updates(self, loss, params):
        # "decrease the learning rate by a factor of 10 after every 30,000 iterations"
        if self.iterations % 30000 == 0:
            self.lr /= 10.
            print('decay LR to: ', self.lr)
        super().get_updates(loss, params)


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

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Dropout(0.5))

    # Regression model
    model.add(Dense(8))

    sgd = SGD(lr=0.005, momentum=0.9, nesterov=True)

    # For a mean squared error regression problem
    model.compile(optimizer=sgd,
                  loss='mean_squared_error')

    plot_model(model, show_shapes=True)


if __name__ == '__main__':
    main()
