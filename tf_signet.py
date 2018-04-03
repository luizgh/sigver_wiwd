import tensorflow as tf
from tensorflow.contrib import slim

from lasagne_to_tf import copy_initializer, transpose_copy_initializer


def build_architecture(input_var, params):
    """ Creates the CNN model described in the paper. Loads the learned weights.
        
        input_var: tf.placeholder of size (None, 150, 220, 1)
        params: the learned parameters

    """
    net = {}
    net['input'] = input_var
    conv1 = conv_bn(input_var, 'conv1',
                    num_outputs=96, kernel_size=11, stride=4,
                    weights=params[0], beta=params[1], gamma=params[2],
                    mean=params[3], inv_std=params[4])
    pool1 = slim.max_pool2d(conv1, 3, 2, scope='pool1')

    conv2 = conv_bn(pool1, 'conv2', num_outputs=256, kernel_size=5, padding='SAME',
                    weights=params[5], beta=params[6], gamma=params[7],
                    mean=params[8], inv_std=params[9])
    pool2 = slim.max_pool2d(conv2, 3, 2, scope='pool2')

    conv3 = conv_bn(pool2, 'conv3', num_outputs=384, kernel_size=3, padding='SAME',
                    weights=params[10], beta=params[11], gamma=params[12],
                    mean=params[13], inv_std=params[14])

    conv4 = conv_bn(conv3, 'conv4', num_outputs=384, kernel_size=3, padding='SAME',
                    weights=params[15], beta=params[16], gamma=params[17],
                    mean=params[18], inv_std=params[19])

    conv5 = conv_bn(conv4, 'conv5', num_outputs=256, kernel_size=3, padding='SAME',
                    weights=params[20], beta=params[21], gamma=params[22],
                    mean=params[23], inv_std=params[24])

    pool5 = slim.max_pool2d(conv5, 3, 2, scope='pool5')

    # Transpose pool5 activations to the lasagne standard, before flattening
    pool5 = tf.transpose(pool5, (0,3,1,2))
    pool5_flat = slim.flatten(pool5)

    net['fc1'] = dense_bn(pool5_flat, 'fc1', 2048,
                   weights=params[25], beta=params[26], gamma=params[27],
                   mean=params[28], inv_std=params[29])

    net['fc2'] = dense_bn(net['fc1'], 'fc2', 2048,
                   weights=params[30], beta=params[31], gamma=params[32],
                   mean=params[33], inv_std=params[34])

    return net


# Helper functions:

def batch_norm(input, scope, beta, gamma, mean, inv_std):
    """ Implements Batch normalization (http://arxiv.org/abs/1502.03167)
        Uses the variables (beta and gamma) learned by the model;
        Uses the statistics (mean, inv_std) collected from training data """
    with tf.name_scope(scope):
        beta_var = tf.Variable(beta, name='beta', dtype=tf.float32)
        gamma_var = tf.Variable(gamma, name='gamma', dtype=tf.float32)
        mean_var = tf.Variable(mean, name='mean', dtype=tf.float32)
        inv_std_var = tf.Variable(inv_std, name='inv_std', dtype=tf.float32)
        return (input - mean_var) * (gamma_var * inv_std_var) + beta_var


def conv_bn(input, scope, num_outputs, kernel_size, weights,
            beta, gamma, mean, inv_std, stride=1, padding='VALID'):
    """ Performs 2D convolution followed by batch normalization and ReLU.
        Uses weigths learned by the model trained using lasagne (transposes them
           to the TensorFlow standard)"""
    conv = slim.conv2d(input, num_outputs=num_outputs, kernel_size=kernel_size,
                      stride=stride, padding=padding, scope=scope,
                      weights_initializer=transpose_copy_initializer(weights),
                      biases_initializer=None, # No biases since we use BN
                      activation_fn=None) # ReLU is applied after BN
    bn = batch_norm(conv, scope='%s_bn' % scope,
                    beta=beta, gamma=gamma, mean=mean, inv_std=inv_std)
    relu = tf.nn.relu(bn)
    return relu


def dense_bn(input, scope, num_outputs, weights, beta, gamma, mean, inv_std):
    """ Implements a fully connected layer followed by batch normalization and
        ReLU """
    with tf.variable_scope(scope):
        w = tf.Variable(weights, name='w', dtype=tf.float32)
        dense = tf.matmul(input, w)
    bn = batch_norm(dense, scope='%s_bn' % scope,
                    beta=beta, gamma=gamma, mean=mean, inv_std=inv_std)
    relu = tf.nn.relu(bn)
    return relu
