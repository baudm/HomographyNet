#!/usr/bin/env python

import os.path
import sys

from keras.models import load_model

from homographynet import data
from homographynet.losses import euclidean_distance


def main():
    if len(sys.argv) != 2:
        name = os.path.basename(__file__)
        print('Usage: {} [trained model.hdf5]'.format(name))
        exit(1)

    model = load_model(sys.argv[1], compile=False)
    batch_size = 64 * 2

    loader = data.loader(data.TEST_PATH, batch_size)
    kwargs = data.get_evaluate_generator_kwargs(batch_size)

    # Optimizer doesn't matter in this case
    model.compile('sgd', euclidean_distance, metrics=['mean_absolute_error'])
    evaluation = model.evaluate_generator(loader, **kwargs)
    print('Test loss:', evaluation)


if __name__ == '__main__':
    main()
