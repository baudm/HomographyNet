#!/usr/bin/env python

import os.path
import sys

import math

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD

from homographynet import data
from homographynet.callbacks import LearningRateScheduler
from homographynet.losses import mean_corner_error
from homographynet.models import create_model


def main():
    if len(sys.argv) > 2:
        name = os.path.basename(__file__)
        print('Usage: {} [existing model.h5]'.format(name))
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

    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=[mean_corner_error])
    model.summary()

    save_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = ModelCheckpoint(os.path.join(save_path, 'model.{epoch:02d}.h5'))

    # LR scaling as described in the paper
    lr_scheduler = LearningRateScheduler(base_lr, 0.1, 30000)

    # In the paper, the 90,000 iterations was for batch_size = 64
    # So scale appropriately
    target_iterations = int(target_iterations * 64 / batch_size)
    # As stated in Keras docs
    steps_per_epoch = int(data.TRAIN_SAMPLES / batch_size)
    epochs = int(math.ceil(target_iterations / steps_per_epoch))

    loader = data.loader(data.TRAIN_PATH, batch_size)

    val_loader = data.loader(data.TEST_PATH, batch_size)
    val_steps = int(data.TEST_SAMPLES / batch_size)

    # Train
    model.fit_generator(loader, steps_per_epoch, epochs,
                        callbacks=[lr_scheduler, checkpoint],
                        validation_data=val_loader, validation_steps=val_steps)


if __name__ == '__main__':
    main()
