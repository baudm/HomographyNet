#!/usr/bin/env python

import os.path
import sys

from keras.models import load_model

from homographynet import data
from homographynet.models import create_model
from homographynet.losses import mean_corner_error


def main():
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [trained model.h5]'.format(name))
        exit(1)

    if len(sys.argv) == 2:
        model = load_model(sys.argv[1], compile=False)
    else:
        model = create_model(use_weights=True)

    model.summary()

    batch_size = 64 * 2

    loader = data.loader(data.TEST_PATH, batch_size)
    steps = int(data.TEST_SAMPLES / batch_size)

    # Optimizer doesn't matter in this case, we just want to set the loss and metrics
    model.compile('sgd', loss='mean_squared_error', metrics=[mean_corner_error])
    evaluation = model.evaluate_generator(loader, steps)
    print('Test loss:', evaluation)


if __name__ == '__main__':
    main()
