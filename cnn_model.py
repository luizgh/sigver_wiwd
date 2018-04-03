from six.moves import cPickle
import lasagne
import theano
from theano import tensor as T
import numpy as np
import six

class CNNModel:
    """ Represents a model trained with the Lasagne library.
    """

    def __init__(self, model_factory, model_weight_path):
        """ Loads the CNN model

        Parameters:
            model_factory (module): An object containing a
                    "build_architecture"function.
            model_weights_path (str): A file containing the trained weights
        """
        with open(model_weight_path, 'rb') as f:
            if six.PY2:
                model_params = cPickle.load(f)
            else:
                model_params = cPickle.load(f, encoding='latin1')

        self.input_size = model_params['input_size']
        self.img_size = model_params['img_size']

        net_input_size = (None, 1, self.input_size[0], self.input_size[1])
        self.model = model_factory.build_architecture(net_input_size,
                                                      model_params['params'])

        self.forward_util_layer = {}  # Used for caching the functions

    def get_feature_vector(self, image, layer='fc2'):
        """ Runs forward propagation until a desired layer, for one input image

        Parameters:
            image (numpy.ndarray): The input image
            layer (str): The desired output layer

        """

        assert len(image.shape) == 2, "Input should have two dimensions: H x W"

        input = image[np.newaxis, np.newaxis]

        # Cache the function that performs forward propagation to the desired layer
        if layer not in self.forward_util_layer:
            inputs = T.tensor4('inputs')
            outputs = lasagne.layers.get_output(self.model[layer],
                                                inputs=inputs,
                                                deterministic=True)
            self.forward_util_layer[layer] = theano.function([inputs], outputs)

        # Perform forward propagation until the desired layer
        out = self.forward_util_layer[layer](input)
        return out

    def get_feature_vector_multiple(self, images, layer='fc2'):
        """ Runs forward propagation until a desired layer, for one input image

        Parameters:
            images (numpy.ndarray): The input images. Should have three dimensions:
                    N x H x W, where N: number of images, H: height, W: width
            layer (str): The desired output layer

        """

        images = np.asarray(images)
        assert len(images.shape) == 3, "Input should have three dimensions: N x H x W"

        # Add the "channel" dimension:
        input = np.expand_dims(images, axis=1)

        # Cache the function that performs forward propagation to the desired layer
        if layer not in self.forward_util_layer:
            inputs = T.tensor4('inputs')
            outputs = lasagne.layers.get_output(self.model[layer],
                                                inputs=inputs,
                                                deterministic=True)
            self.forward_util_layer[layer] = theano.function([inputs], outputs)

        # Perform forward propagation until the desired layer
        out = self.forward_util_layer[layer](input)
        return out
