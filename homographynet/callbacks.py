#!/usr/bin/env python

from keras.callbacks import Callback
import keras.backend as K


class LearningRateScheduler(Callback):
    """Learning rate scheduler.

    See Caffe SGD docs
    """

    def __init__(self, base_lr, gamma, step_size):
        super().__init__()
        self._base_lr = base_lr
        self._gamma = gamma
        self._step_size = step_size
        self._steps = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._steps = epoch * self.params['steps']

    def on_batch_begin(self, batch, logs=None):
        self._steps += 1
        if self._steps % self._step_size == 0:
            exp = int(self._steps / self._step_size)
            lr = self._base_lr * (self._gamma ** exp)
            K.set_value(self.model.optimizer.lr, lr)
            print('New learning rate:', lr)
