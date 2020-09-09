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

#%%
import os, glob, shutil as shu
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans


#%%
mode = "-mac"  # TODO: use argparse instead!!

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

#%%
# * 1. Inter-WSI cross validation

# * 1. 选择8个做training/validation，1个做test
""" 
* 1 获取编号列表
* 2 取8个
* 3 在每个子文件夹中都取这8个片子
    DAB--1~8
    Mask--1~8
    ...
* 4 Write a generator to return 写生成器返回：
    DAB
        WSI1
            Tiles
                Tumor
                    <All 10 folds>
                        * ! images------.
    Chips ------------------------------|
        WSI1                            |
            Tiles                       |
                Tumor                   |
                    <All 10 folds>      |
                        * ! images -----.------- Zipped one by one as tuples
"""


def kmrGenerator(
    batch_size,
    dataset_path,
    image_folder,
    mask_folder,
    aug_dict,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    flag_multi_class=False,
    num_class=2,
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
):
    train_val_datagen = ImageDataGenerator(**aug_dict)
    test_datagen = ImageDataGenerator(**aug_dict)

    train_val_path = ""
    test_path = ""

    train_val_generator = train_val_datagen.flow_from_directory(
        dataset_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    
    pass


#%%
# * 2. Cross validation
# %%
