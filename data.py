#!/usr/bin/env python

import os.path
import glob

import numpy as np


_SAMPLES_PER_ARCHIVE = 7680

TRAIN_PATH = '/home/darwin/Projects/HomographyNet/repack'
TRAIN_SAMPLES = 65 * _SAMPLES_PER_ARCHIVE

TEST_PATH = '/home/darwin/Projects/HomographyNet/test-set'
TEST_SAMPLES = 5 * _SAMPLES_PER_ARCHIVE


def get_fit_generator_kwargs(batch_size, target_iterations):
    # In the paper, the 90,000 iterations was for batch_size = 64
    # So scale appropriately
    total_iterations = int(target_iterations * 64 / batch_size)
    # As stated in Keras docs
    steps_per_epoch = int(TRAIN_SAMPLES / batch_size)
    epochs = int(np.ceil(total_iterations / steps_per_epoch))
    # kwargs for model.fit_generator()
    return {'steps_per_epoch': steps_per_epoch, 'epochs': epochs}


def get_evaluate_generator_kwargs(batch_size):
    steps = int(TEST_SAMPLES / batch_size)
    return {'steps': steps}


def loader(path, batch_size=64, normalize=True):
    """Generator to be used with model.fit_generator()"""
    while True:
        for npz in glob.glob(os.path.join(path, '*.npz')):
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            del archive
            # Split into mini batches
            num_batches = int(len(offsets) / batch_size)
            images = np.array_split(images, num_batches)
            offsets = np.array_split(offsets, num_batches)
            while offsets:
                batch_images = images.pop()
                batch_offsets = offsets.pop()
                if normalize:
                    batch_images = (batch_images - 127.5) / 127.5
                    batch_offsets = batch_offsets / 32.
                yield batch_images, batch_offsets
