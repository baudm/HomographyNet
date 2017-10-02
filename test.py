#!/usr/bin/env python

import sys
import os.path

from keras.models import load_model

import data


def main():
    if len(sys.argv) != 2:
        name = os.path.basename(__file__)
        print('Usage: {} [trained model.hdf5]'.format(name))
        exit(1)

    model = load_model(sys.argv[1])
    batch_size = 64 * 2

    loader = data.loader(data.TEST_PATH, batch_size)
    kwargs = data.get_evaluate_generator_kwargs(batch_size)

    evaluation = model.evaluate_generator(loader, **kwargs)
    print('Test loss:', evaluation)


if __name__ == '__main__':
    main()
