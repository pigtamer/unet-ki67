""" Dealing with the data from Kimura

* Plan 1.

Inter-WSI validation
Do not use the folds inside of each WSI.
We have 9 WSIs. Presume that chips in one case should not be mixed together,
then inter-WSI validation is to choose k WSIs for training and 9-k WSIs for validation at a time.


* Plan 2.

Cross validation.
Use folds inside each WSI.
Now we created 10 folds. k folds (say 9) from ALL WSIs are involved in training
    and 10-k folds are involved in validation at a time.

"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans

mode = "mac"  # TODO: use argparse instead!!

if mode == "mac":
    # use plaidml backend for Mac OS
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator


""" 
* 目录结构

TILES_(256, 256)/
├── DAB
│   ├── 01_14-3768_Ki67_HE
│   │   ├── Annotations
│   │   │   └── annotations-01_14-3768_Ki67_HE.txt
│   │   └── Tiles
│   │       ├── Healthy Tissue
│   │       │   ├── 001 [3328 entries 
...

 """

# * 1. Inter-WSI cross validation


# * 2. Cross validation