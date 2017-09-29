#!/usr/bin/env python

import os.path
import glob

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Dropout, BatchNormalization, Flatten
from keras.optimizers import SGD as _SGD
from keras.callbacks import ModelCheckpoint
#from keras.models import load_model
#from keras.utils.vis_utils import plot_model


class SGD(_SGD):
    """
    SGD optimizer with LR decay logic similar to the Caffe implementation

    To use properly, make sure that decay is always set to 0.
    """
    def get_updates(self, loss, params):
        # "decrease the learning rate by a factor of 10 after every 30,000 iterations"
        if self.iterations % 30000 == 0:
            self.lr *= 0.1
            print('decay LR to: ', self.lr)
        # This method is decorated so just call it directly
        _SGD.get_updates(self, loss, params)


def data_loader(path, batch_size=64):
    """Generator to be used with model.fit_generator()"""
    while True:
        for npz in glob.glob(os.path.join(path, '*.npz')):
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            # Normalize
            images = (images - 127.5) / 127.5
            offsets = offsets / 32.
            # Yield minibatch
            for i in range(0, len(offsets), batch_size):
                end_i = i + batch_size
                try:
                    batch_images = images[i:end_i]
                    batch_offsets = offsets[i:end_i]
                except IndexError:
                    continue
                yield batch_images, batch_offsets


def main():
    model = Sequential()
    model.add(InputLayer((128, 128, 2)))

    # 4 Layers with 64 filters, then another 4 with 128 filters
    filters = 4 * [64] + 4 * [128]
    for i, f in enumerate(filters, 1):
        model.add(Conv2D(f, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        # MaxPooling after every 2 Conv layers except the last one
        if i % 2 == 0 and i != 8:
            model.add(MaxPooling2D(strides=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    # Regression model
    model.add(Dense(8))

    sgd = _SGD(lr=0.005, momentum=0.9)

    # For a mean squared error regression problem
    model.compile(optimizer=sgd,
                  loss='mse')

    #plot_model(model, show_shapes=True)
    model.summary()
    
    checkpoint = ModelCheckpoint('/home/darwin/Projects/HomographyNet/model.{epoch:02d}.hdf5')

    # Configuration
    data_path = '/home/darwin/Projects/datasets/packed'
    test_path = '/home/darwin/Projects/datasets/hey'
    num_samples = 150 * 3072 # 158 archives x 3,072 samples per archive
    batch_size = 64
    total_iterations = 90000

    steps_per_epoch = num_samples / batch_size # As stated in Keras docs
    epochs = int(total_iterations / steps_per_epoch) # Use 90,000 total iterations like in the paper

    # model is some Keras Model instance
    model.fit_generator(data_loader(data_path, batch_size),
                        callbacks=[checkpoint],
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs)
    #model = load_model('model.12.hdf5')
    # Test
    evaluation = model.evaluate_generator(data_loader(test_path, 1), steps=5000)
    print(evaluation)
    x,y = next(data_loader(test_path, 1))
    print(model.predict(x, batch_size=1, verbose=1))
    print(y)
    

if __name__ == '__main__':
    main()
