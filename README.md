# HomographyNet
Implementation of HomographyNet in Keras

## Project Organization
### dataset
* Contains `generate.py`, a script for generating a suitable dataset for the HomographyNet model.
* A Jupyter notebook with general info about the structure of the generated dataset.

### demo
* Jupyter notebook with the training/test results
* Raw image files used
* Utility code used in the demo
* Training history in JSON format 

### homographynet
A Python package containing the implementation of the HomographyNet model in Keras.

### models
* `homographynet_weights_tf_dim_ordering_tf_kernels.h5` - pretrained weights of the baseline HomographyNet model
* `mobile_homographynet_weights_tf_dim_ordering_tf_kernels.h5` - pretrained weights of the MobiletNet-based model

`test.py` - a script for evaluating a specified model, or the pretrained model. It will download and cache the weights of the pretrained model on first use.
`train.py` - a script for training the HomographyNet model
