""" This example shows how to extract features for a new signature, 
    using the CNN trained on the GPDS dataset. It also compares the
    results with the ones obtained by the authors, to ensure consistency.

    Note that loading and compiling the model takes time. It is preferable
    to load and process multiple signatures in the same python session.

"""
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import numpy as np

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

processed_correct = np.load('data/processed.npy')

assert np.allclose(processed_correct, processed), "The preprocessed image is different than expected. "+ \
                                                 "Check the version of packages 'scipy' and 'pillow'"

feature_vector_correct = np.load('data/some_signature_signet.npy')
assert np.allclose(feature_vector_correct, feature_vector, atol=1e-3)

print('Tests passed.')
