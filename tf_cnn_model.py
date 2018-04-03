from six.moves import cPickle
import tensorflow as tf
import numpy as np
import six

class TF_CNNModel:
    """ Represents a TF model (in this case, with weights trained with the Lasagne library.)
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

        net_input_size = (None, self.input_size[0], self.input_size[1], 1)
        self.x_input = tf.placeholder(tf.float32, net_input_size)
        self.model = model_factory.build_architecture(self.x_input,
                                                      model_params['params'])
    

    def get_feature_vector(self, sess, image, layer='fc2'):
        """ Runs forward propagation until a desired layer, for one input image

        Parameters:
            sess (tf session)
            image (numpy.ndarray): The input image
            layer (str): The desired output layer

        """

        assert len(image.shape) == 2, "Input should have two dimensions: H x W"

        input = image[np.newaxis, :, :, np.newaxis]

        out = sess.run(self.model[layer], feed_dict={self.x_input: input})
        return out

    def get_feature_vector_multiple(self, sess, images, layer='fc2'):
        """ Runs forward propagation until a desired layer, for one input image

        Parameters:
            images (numpy.ndarray): The input images. Should have three dimensions:
                    N x H x W, where N: number of images, H: height, W: width
            layer (str): The desired output layer

        """

        images = np.asarray(images)
        assert len(images.shape) == 3, "Input should have three dimensions: N x H x W"

        # Add the "channel" dimension:
        input = np.expand_dims(images, axis=3)


        # Perform forward propagation until the desired layer
        out = sess.run(self.model[layer], feed_dict={self.x_input: input})
        return out
