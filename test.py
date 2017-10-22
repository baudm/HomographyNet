#!/usr/bin/env python

import os.path
import sys

from keras.models import load_model
from keras.utils.data_utils import get_file

from homographynet import data
from homographynet.models import create_model
from homographynet.losses import mean_corner_error


WEIGHTS_NAME = 'homographynet_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH = 'https://github.com/baudm/HomographyNet/raw/master/models/' + WEIGHTS_NAME


def load_pretrained_model():
    weights_path = get_file(WEIGHTS_NAME, WEIGHTS_PATH,
                            cache_subdir='models',
                            md5_hash='3118ab8ddb49dfa48b38d7cad7efcb88')
    model = create_model()
    model.load_weights(weights_path)
    return model


def main():
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [trained model.h5]'.format(name))
        exit(1)

    if len(sys.argv) == 2:
        model = load_model(sys.argv[1], compile=False)
    else:
        model = load_pretrained_model()


    batch_size = 64 * 2

    loader = data.loader(data.TEST_PATH, batch_size)
    steps = int(data.TEST_SAMPLES / batch_size)

    # Optimizer doesn't matter in this case, we just want to set the loss and metrics
    model.compile('sgd', loss='mean_squared_error', metrics=[mean_corner_error])
    evaluation = model.evaluate_generator(loader, steps)
    print('Test loss:', evaluation)


if __name__ == '__main__':
    main()
