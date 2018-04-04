from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.layers import InputLayer, DenseLayer, batch_norm
from lasagne.layers import SpatialPyramidPoolingLayer
import lasagne


def build_architecture(input_shape, trained_weights=None):
    net = {}

    net['input'] = InputLayer((None,1,None,None))
    net['large_conv1'] = batch_norm(Conv2DLayer(net['input'], num_filters=32, filter_size=11, stride=3, pad=5, flip_filters=False))
    net['large_pool1'] = MaxPool2DLayer(net['large_conv1'], pool_size=3, stride=2)

    net['large_conv2'] = batch_norm(Conv2DLayer(net['large_pool1'], num_filters=64, filter_size=5, pad=2, flip_filters=False))
    net['large_pool2'] = MaxPool2DLayer(net['large_conv2'], pool_size=3)

    net['large_conv3'] = batch_norm(Conv2DLayer(net['large_pool2'], num_filters=128, filter_size=3, pad=1, flip_filters=False))
    net['large_conv4'] = batch_norm(Conv2DLayer(net['large_conv3'], num_filters=128, filter_size=3, pad=1, flip_filters=False))
    net['large_pool4'] = MaxPool2DLayer(net['large_conv4'], pool_size=2)
    net['large_conv5'] = batch_norm(Conv2DLayer(net['large_pool4'], num_filters=128, filter_size=3, pad=1, flip_filters=False))

    net['large_pool5'] = SpatialPyramidPoolingLayer(net['large_conv5'], implementation='kaiming')

    net['fc1'] = batch_norm(DenseLayer(net['large_pool5'], num_units=2048))
    net['fc2'] = batch_norm(DenseLayer(net['fc1'], num_units=2048))

    if trained_weights:
        lasagne.layers.set_all_param_values(net['fc2'], trained_weights)

    return net
