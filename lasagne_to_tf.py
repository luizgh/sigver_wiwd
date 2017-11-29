""" Some useful functions to port a model from lasagne to tensorflow.

    * Lasagne uses the format BCHW, while tensorflow uses BHWC 
      (B = batch_size, C = channels, H = height, W = width)
    * By default, lasagne uses convolution, while tensorflow implements
      cross-correlation (convolution is equivalent to cross-correlation with flipped filters)

    Here we define some functions to change the filters from one format to the other
"""

import numpy as np

class copy_initializer:
    def __init__(self, value_to_copy):
        self.value_to_copy = value_to_copy

    def __call__(self, shape, **kwargs):
        expected_shape = list(shape)
        actual_shape = list(self.value_to_copy.shape)
        assert actual_shape == expected_shape, 'Invalid shape for initilizer. Expected: %s. Given: %s.' % (expected_shape, actual_shape)
        return self.value_to_copy

class flipping_copy_initializer (copy_initializer):
    def __init__(self, value_to_copy):
        v = np.transpose(value_to_copy, [2,3,1,0])
        v = v [::-1,::-1,:,:]
        self.value_to_copy = v

class transpose_copy_initializer (copy_initializer):
    def __init__(self, value_to_copy):
        v = np.transpose(value_to_copy, [2,3,1,0])
        self.value_to_copy = v
