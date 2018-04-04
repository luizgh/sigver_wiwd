# Learned representation for Offline Handwritten Signature Verification

This repository contains the code and instructions to use the trained CNN models described in [1] to extract features for Offline Handwritten Signatures. 
It also includes the models described in [2] that can generate a fixed-sized feature vector for signatures of different sizes. 

[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))

[2] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Fixed-sized representation learning from Offline Handwritten Signatures of different sizes" ([preprint](https://arxiv.org/abs/1804.00448))

Topics:

* [Installation](#installation): How to set-up the dependencies / download the models to extract features from new signatures
* [Usage](#usage): How to use this code as a feature extractor for signature images
* [Using the features in Matlab](#using-the-features-in-matlab): A script to facilitate processing multiple signatures and saving the features in matlab (.mat) format
* [Datasets](#datasets): Download extracted features (using the proposed models) for the GPDS, MCYT, CEDAR and Brazilian PUC-PR datasets (for the methods presented in [1] - .mat files that do not require any pre-processing code)


# Installation

## Pre-requisites 

The code is written in Python 2<sup>1</sup>. We recommend using the Anaconda python distribution ([link](https://www.continuum.io/downloads)), and create a new environment using: 
```
conda create -n sigver -y python=2
source activate sigver
```

The following libraries are required

* Scipy version 0.18
* Pillow version 3.0.0
* OpenCV
* Theano<sup>2</sup>
* Lasagne<sup>2</sup>

They can be installed by running the following commands: 

```
conda install -y "scipy=0.18.0" "pillow=3.0.0"
conda install -y jupyter notebook matplotlib # Optional, to run the example in jupyter notebook
pip install opencv-python
pip install "Theano==0.9"
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

We tested the code in Ubuntu 16.04. This code can be used with or without GPUs - to use a GPU with Theano, follow the instructions in this [link](http://deeplearning.net/software/theano/tutorial/using_gpu.html). Note that Theano takes time to compile the model, so it is much faster to instantiate the model once and run forward propagation for many images (instead of calling many times a script that instantiates the model and run forward propagation for a single image).

<sup>1</sup> Python 3.5 can be also be used, but the feature vectors will differ from those generated from Python 2 (due to small differences in preprocessing the images). Either version can be used, but feature vectors generated from different versions should not be mixed. Note that the data on section [Datasets](#datasets) has been obtained using Python 2. 

<sup>2</sup> Although we used Theano and Lasagne for training, you can also use TensorFlow to extract the features. See tf_example.py for details.

## Downloading the models

* Clone (or download) this repository
* Download the pre-trained models from the [project page](https://www.etsmtl.ca/Unites-de-recherche/LIVIA/Recherche-et-innovation/Projets/Signature-Verification)
  * Save / unzip the models in the "models" folder

Or simply run the following to download both the SigNet models(from [1]) and SigNet-SPP models (from [2]): 
```
git clone https://github.com/luizgh/sigver_wiwd.git
cd sigver_wiwd/models
wget "https://storage.googleapis.com/luizgh-datasets/models/signet_models.zip"
wget "https://storage.googleapis.com/luizgh-datasets/models/signet_spp_models.zip"
unzip signet_models.zip
unzip signet_spp_models.zip
``` 

## Testing 

Run ```python example.py``` and ```python example_spp.py```. These scripts pre-process a signature, and compare the feature vectors obtained by the model to the results obtained by the author. If the test fails, please check the versions of Scipy and Pillow. I noticed that different versions of these libraries produce slightly different results for the pre-processing steps.

# Usage

The following code (from example.py) shows how to load, pre-process a signature, and extract features using one of the learned models:

```python
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import signet
from cnn_model import CNNModel

# Maximum signature size (required for the SigNet models):
canvas_size = (952, 1360)  

# Load and pre-process the signature
original = imread('data/some_signature.png', flatten=1)

processed = preprocess_signature(original, canvas_size)

# Load the model
model_weight_path = 'models/signet.pkl'
model = CNNModel(signet, model_weight_path)

# Use the CNN to extract features
feature_vector = model.get_feature_vector(processed)

# Multiple images can be processed in a single forward pass using:
# feature_vectors = model.get_feature_vector_multiple(images)
```

Note that for the SigNet models (from [1]) the signatures used in the ```get_feature_vector``` method must always have the same size as those used for training the system (150 x 220 pixels). 

For the SigNet-SPP methods (from [2]) the signatures can have any size. We provide models trained on signatures scanned at 300dpi and signatures scanned at 600dpi. Refer to the paper for more details on this method.

For an interactive example, use jupyter notebook:
```
jupyter notebook
```

Look for the notebook "interactive_example.ipynb". You can also visualize it directly [here](https://github.com/luizgh/sigver_wiwd/blob/master/interactive_example.ipynb)


## Using the features in Matlab

While the code requires python (with the libraries mentioned above) to extract features, it is possible to save the results in a matlab format. We included a script that process all signatures in a folder and save the results in matlab files (one .mat file for each signature). 

Usage: 
```
python process_folder.py <signatures_path> <save_path> <model_path> [canvas_size]
```

Example:
```
python process_folder.py signatures/ features/ models/signet.pkl
```

This will process all signatures in the "signatures" folder, using the SigNet model, and save one .mat file in the folder "features" for each signatures. Each file contains a single variable named "feature_vector" with the features extracted from the signature.

# Datasets

To faciliate further research, we are also making available the features extracted for each of the four datasets used in this work (GPDS, MCYT, CEDAR, Brazilian PUC-PR), using the models SigNet and SigNet-F (with lambda=0.95).

 |Dataset | SigNet | SigNet-F |
 | --- | --- | --- |
 | GPDS | [GPDS_signet](https://storage.googleapis.com/luizgh-datasets/datasets/gpds_signet.zip) | [GPDS_signet_f](https://storage.googleapis.com/luizgh-datasets/datasets/gpds_signet_f.zip) |
| MCYT | [MCYT_signet](https://storage.googleapis.com/luizgh-datasets/datasets/mcyt_signet.zip) | [MCYT_signet_f](https://storage.googleapis.com/luizgh-datasets/datasets/mcyt_signet_f.zip) |
| CEDAR | [CEDAR_signet](https://storage.googleapis.com/luizgh-datasets/datasets/cedar_signet.zip) | [CEDAR_signet_f](https://storage.googleapis.com/luizgh-datasets/datasets/cedar_signet_f.zip) |
| Brazilian PUC-PR\* | [brazilian_signet](https://storage.googleapis.com/luizgh-datasets/datasets/brazilian_signet.zip) | [brazilian_signet_f](https://storage.googleapis.com/luizgh-datasets/datasets/brazilian_signet_f.zip) |

There are two files for each user: real_X.mat and forg_X.mat. The first contains a matrix of size N x 2048, containing the feature vectors of N genuine signatures from that user. The second contains a matrix of size M x 2048, containing the feature vectors of each of the M skilled forgeries made targetting the user. 

\* Note: for the brazilian PUC-PR dataset, the first 10 forgeries are "Simple forgeries", while the last 10 forgeries are "Skilled forgeries".

## Loading the feature vectors in matlab

```
f = load('real_2.mat')
% f.features: [Nx2048 single]
```

## Loading the feature vectors in python

```
from scipy.io import loadmat
features = loadmat('real_2.mat')['features']
# features: numpy array of shape (M, 2048)
```

# Citation

If you use our code, please consider citing the following papers:

[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))

[2] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Fixed-sized representation learning from Offline Handwritten Signatures of different sizes" ([preprint](https://arxiv.org/abs/1804.00448))

If using any of the four datasets mentioned above, please cite the paper that introduced the dataset:

GPDS: Vargas, J.F., M.A. Ferrer, C.M. Travieso, and J.B. Alonso. 2007. “Off-Line Handwritten Signature GPDS-960 Corpus.” In Document Analysis and Recognition, 9th I    nternational Conference on, 2:764–68. doi:10.1109/ICDAR.2007.4377018.

MCYT: Ortega-Garcia, Javier, J. Fierrez-Aguilar, D. Simon, J. Gonzalez, M. Faundez-Zanuy, V. Espinosa, A. Satue, et al. 2003. “MCYT Baseline Corpus: A Bimodal Biometric Database.” IEE Proceedings-Vision, Image and Signal Processing 150 (6): 395–401.

CEDAR: Kalera, Meenakshi K., Sargur Srihari, and Aihua Xu. 2004. “Offline Signature Verification and Identification Using Distance Statistics.” International Journal     of Pattern Recognition and Artificial Intelligence 18 (7): 1339–60. doi:10.1142/S0218001404003630.

Brazilian PUC-PR: Freitas, C., M. Morita, L. Oliveira, E. Justino, A. Yacoubi, E. Lethelier, F. Bortolozzi, and R. Sabourin. 2000. “Bases de Dados de Cheques Bancarios Brasilei    ros.” In XXVI Conferencia Latinoamericana de Informatica.

# License

The source code is released under the BSD 2-clause license. Note that the trained models used the GPDS dataset for training (which is restricted for non-comercial use). 
