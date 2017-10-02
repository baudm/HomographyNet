#!/usr/bin/env python

import sys
import os.path

from keras.models import load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import data
from models import create_model
from callbacks import LearningRateScheduler


def main():
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [existing model.hdf5]'.format(name))
        exit(1)

    if len(sys.argv) == 2:
        model = load_model(sys.argv[1], compile=False)
    else:
        model = create_model()

    # Configuration
    batch_size = 64
    target_iterations = 90000 # at batch_size = 64
    base_lr = 0.005

    sgd = SGD(lr=base_lr, momentum=0.9)

    model.compile(optimizer=sgd, loss='mse')
    model.summary()

    save_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = ModelCheckpoint(os.path.join(save_path, 'model.{epoch:02d}.hdf5'))

    # LR scaling as described in the paper
    lr_scheduler = LearningRateScheduler(base_lr, 0.1, 30000)

    loader = data.loader(data.TRAIN_PATH, batch_size)
    kwargs = data.get_fit_generator_kwargs(batch_size, target_iterations)
    # Train
    model.fit_generator(loader, callbacks=[lr_scheduler, checkpoint], **kwargs)


if __name__ == '__main__':
    main()
