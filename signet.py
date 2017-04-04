import lasagne

from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.layers import InputLayer, DenseLayer, batch_norm


def build_architecture(input_shape, trained_weights=None):
    """ Build the Theano symbolic graph representing the CNN model.

    :param input_shape: A tuple representing the input shape (h,w)
    :param trained_weights: Pre-trained weights. If None, the network is initialized at random.
    :return: A dictionary containing all layers
    """
    net = {}

    net['input'] = InputLayer(input_shape)

    net['conv1'] = batch_norm(Conv2DLayer(net['input'], num_filters=96, filter_size=11, stride=4, flip_filters=False))
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=3, stride=2)

    net['conv2'] = batch_norm(Conv2DLayer(net['pool1'], num_filters=256, filter_size=5, pad=2, flip_filters=False))
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=3, stride=2)

    net['conv3'] = batch_norm(Conv2DLayer(net['pool2'], num_filters=384, filter_size=3, pad=1, flip_filters=False))
    net['conv4'] = batch_norm(Conv2DLayer(net['conv3'], num_filters=384, filter_size=3, pad=1, flip_filters=False))
    net['conv5'] = batch_norm(Conv2DLayer(net['conv4'], num_filters=256, filter_size=3, pad=1, flip_filters=False))
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=3, stride=2)

    net['fc1'] = batch_norm(DenseLayer(net['pool5'], num_units=2048))
    net['fc2'] = batch_norm(DenseLayer(net['fc1'], num_units=2048))

    if trained_weights:
        lasagne.layers.set_all_param_values(net['fc2'], trained_weights)

    return net
