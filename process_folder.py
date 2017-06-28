""" This example extract features for all signatures in a folder,
    using the CNN trained on the GPDS dataset. Results are saved in a matlab
    format.

    Usage: python process_folder.py <signatures_path> <save_path>
                                    <model_path> [canvas_size]

    Example:
    python process_folder.py signatures/ features/ models/signet.pkl

    This example will process all signatures in the "signatures" folder, using
    the SigNet model, and saving the resutls to the features folder

"""
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel
import numpy as np
import sys
import os
import scipy.io

if len(sys.argv) not in [4,6]:
    print('Usage: python process_folder.py <signatures_path> <save_path> '
          '<model_path> [canvas_size]')
    exit(1)

signatures_path = sys.argv[1]
save_path = sys.argv[2]
model_path = sys.argv[3]
if len(sys.argv) == 4:
    canvas_size = (952, 1360)  # Maximum signature size
else:
    canvas_size = (int(sys.argv[4]), int(sys.argv[5]))

print('Processing images from folder "%s" and saving to folder "%s"' % (signatures_path, save_path))
print('Using model %s' % model_path)
print('Using canvas size: %s' % (canvas_size,))

# Load the model
model_weight_path = 'models/signet.pkl'
model = CNNModel(signet, model_weight_path)

files = os.listdir(signatures_path)

# Note: it there is a large number of signatures to process, it is faster to
# process them in batches (i.e. use "get_feature_vector_multiple")
for f in files:
    # Load and pre-process the signature
    filename = os.path.join(signatures_path, f)
    original = imread(filename, flatten=1)
    processed = preprocess_signature(original, canvas_size)

    # Use the CNN to extract features
    feature_vector = model.get_feature_vector(processed)

    # Save in the matlab format
    save_filename = os.path.join(save_path, os.path.splitext(f)[0] + '.mat')
    scipy.io.savemat(save_filename, {'feature_vector':feature_vector})
