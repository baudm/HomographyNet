#!/usr/bin/env python

from keras.callbacks import Callback
import keras.backend as K


class LearningRateScheduler(Callback):
    """Learning rate scheduler.

    See Caffe SGD docs
    """

    def __init__(self, base_lr, gamma, step_size):
        super().__init__()
        self._lr = base_lr
        self._gamma = gamma
        self._step_size = step_size
        self._iteration = 1

    def on_batch_begin(self, batch, logs=None):
        if self._iteration % self._step_size == 0:
            self._lr *= self._gamma
            K.set_value(self.model.optimizer.lr, self._lr)
            print('New learning rate:', self._lr)
        self._iteration += 1
