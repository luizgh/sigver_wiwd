""" This example shows how to extract features for a new signature, 
    using the CNN trained on the GPDS dataset [1]. It also compares the
    results with the ones obtained by the authors, to ensure consistency.

    Note that loading and compiling the model takes time. It is preferable
    to load and process multiple signatures in the same python session.

    [1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features
    for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks"

"""
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import numpy as np
import six

canvas_size = (952, 1360)  # Maximum signature size

# Load and pre-process the signature
original = imread('data/some_signature.png', flatten=1)

processed = preprocess_signature(original, canvas_size)

# Load the model
model_weight_path = 'models/signet.pkl'
model = CNNModel(signet, model_weight_path)

# Use the CNN to extract features
feature_vector = model.get_feature_vector(processed)

# Compare the obtained feature vector to the expected value 
# (to confirm same results obtained by the authors)

if six.PY2:
    # Note: pre-processing gives slightly different results on Py2 and Py3 (due to
    # changes in scipy and rounding differences between Py2 and Py3). We have different
    # expected results for the two python versions
    processed_correct = np.load('data/processed.npy')
    feature_vector_correct = np.load('data/some_signature_signet.npy')
else:
    processed_correct = np.load('data/processed_py3.npy')
    feature_vector_correct = np.load('data/some_signature_signet_py3.npy')

assert np.allclose(processed_correct, processed), "The preprocessed image is different than expected. "+ \
                                                 "Check the version of packages 'scipy' and 'pillow'"


assert np.allclose(feature_vector_correct, feature_vector, atol=1e-3)

print('Tests passed.')
