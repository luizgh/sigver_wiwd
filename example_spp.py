""" This example shows how to extract features for a new signature,
    using the CNN trained on the GPDS dataset using Spatial Pyramid Pooling[1].
    It also compares the results with the ones obtained by the authors, to
    ensure consistency.

    Note that loading and compiling the model takes time. It is preferable
    to load and process multiple signatures in the same python session.

    [1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Fixed-sized
    representation learning from Offline Handwritten Signatures of different sizes"

"""

from scipy.misc import imread
from preprocess.normalize import remove_background
import signet_spp_300dpi
from cnn_model import CNNModel
import numpy as np
import six

# Load and pre-process the signature
original = imread('data/some_signature.png', flatten=1)

# For the SPP models, signatures from any size can be used. In our experiments
# the best results were obtained padding smaller images (up to a
# standard "canvas size" used for training), and processing larger images
# in their original size. See the paper [1] for more details.

# Even if we are not padding the images, we still need to invert them (0=white, 255=black)
processed = 255 - remove_background(original)

# Load the model
model_weight_path = 'models/signet_spp_300dpi.pkl'
model = CNNModel(signet_spp_300dpi, model_weight_path)

# Use the CNN to extract features
feature_vector = model.get_feature_vector(processed)

# Compare the obtained feature vector to the expected value
# (to confirm same results obtained by the authors)

if six.PY2:
    # Note: pre-processing gives slightly different results on Py2 and Py3 (due to
    # changes in scipy and rounding differences between Py2 and Py3). We have different
    # expected results for the two python versions
    processed_correct = np.load('data/processed_spp.npy')
    feature_vector_correct = np.load('data/some_signature_signet_spp_300dpi.npy')
else:
    processed_correct = np.load('data/processed_spp_py3.npy')
    feature_vector_correct = np.load('data/some_signature_signet_spp_300dpi_py3.npy')

assert np.allclose(processed_correct, processed), "The preprocessed image is different than expected. "+ \
                                                 "Check the version of packages 'scipy' and 'pillow'"

assert np.allclose(feature_vector_correct, feature_vector, atol=1e-3)

print('Tests passed.')
