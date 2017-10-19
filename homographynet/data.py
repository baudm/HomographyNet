#!/usr/bin/env python

import os.path
import glob

import numpy as np


_SAMPLES_PER_ARCHIVE = 7680

TRAIN_PATH = '/home/darwin/Projects/HomographyNet/repack'
TRAIN_SAMPLES = 65 * _SAMPLES_PER_ARCHIVE

TEST_PATH = '/home/darwin/Projects/HomographyNet/test-set'
TEST_SAMPLES = 5 * _SAMPLES_PER_ARCHIVE


def _shuffle_in_unison(a, b):
    """A hack to shuffle both a and b the same "random" way"""
    prng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(prng_state)
    np.random.shuffle(b)


def loader(path, batch_size=64, normalize=True):
    """Generator to be used with model.fit_generator()"""
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            del archive
            _shuffle_in_unison(images, offsets)
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
