""" Dealing with the data from Kimura

* Plan 1. [OK]

Inter-WSI validation
Do not use the folds inside of each WSI.
We have 9 WSIs. Presume that chips in one case should not be mixed together,
then inter-WSI validation is to choose k WSIs for training and 9-k WSIs for validation at a time.


* Plan 2.

Cross validation.
Use folds inside each WSI.
Now we created 10 folds. k folds (say 9) from ALL WSIs are involved in training
    and 10-k folds are involved in validation at a time.

* Plan 3.
Data augmentation and preprocessing with TF generic API
--> i.e. use png/jpeg decoding
"""

#%%
import os, glob, shutil as shu
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans

from data import adjustData
from model import *
from utils import *

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
from itertools import combinations as comb

# %%
# * 3. Tf.data as input pipeline
def load_kmr_tfdata(
    dataset_path,
    batch_size=16,
    wsi_ids=None,
    cross_fold=None,
    stains=["HE", "Mask"],
    aug=False,
    target_size=(256, 256),
    seed=1,
    cache=None,
    shuffle_buffer_size=128,
) -> tuple:
    def parse_image(file_path):
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_png(img, channels=3)

        img = tf.image.convert_image_dtype(img, tf.float32)

        img = tf.image.resize(img, [target_size[0], target_size[1]])
        # resize the image to the desired size.
        if aug:
            img = augment(img)
        return img

    def prepare_for_training(
        ds, cache=cache, shuffle_buffer_size=shuffle_buffer_size, batch_size=batch_size
    ):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(
            buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=False
        )
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    data_generator = {}
    for staintype in stains:
        if staintype != "Mask":

            def augment(image, seed=seed):
                # Add 6 pixels of padding
                image = tf.image.resize_with_crop_or_pad(
                    image, target_size[0] + 32, target_size[0] + 32
                )
                # Random crop back to the original size
                image = tf.image.random_crop(
                    image, size=[target_size[0], target_size[0], 3], seed=seed
                )
                image = tf.image.random_brightness(
                    image, max_delta=0.01, seed=seed
                )  # Random brightness
                image = tf.image.random_flip_left_right(image, seed=seed)
                image = tf.image.random_flip_up_down(image, seed=seed)
                return image

        else:
            # only the channels of input are different
            def augment(image, seed=seed):
                # Add 6 pixels of padding
                image = tf.image.resize_with_crop_or_pad(
                    image, target_size[0] + 32, target_size[0] + 32
                )
                # Random crop back to the original size
                image = tf.image.random_crop(
                    image, size=[target_size[0], target_size[0], 1], seed=seed
                )

                image = tf.image.random_flip_left_right(image, seed=seed)
                image = tf.image.random_flip_up_down(image, seed=seed)
                return image

        dir_pattern = [
            dataset_path
            + "/"
            + staintype
            + "/"
            + wsi
            + "*/Tiles/Tumor/"
            + foldnum
            + "/*"
            for wsi in wsi_ids
            for foldnum in cross_fold
        ]

        list_ds = tf.data.Dataset.list_files(dir_pattern, shuffle=True, seed=seed)
        # list_ds = list_ds.shard(num_shards=hvd.size(), index=hvd.rank())
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        labeled_ds = list_ds.map(parse_image, num_parallel_calls=AUTOTUNE)

        data_generator[staintype] = prepare_for_training(
            labeled_ds,
            cache=(cache + "_%s_%d.tfcache" % (staintype, 1e10 * np.random.rand()))
            if isinstance(cache, str)
            else cache,
        )
    train_generator = zip(data_generator["HE"], data_generator["Mask"])
    n = len(list_ds)
    return (train_generator, n)
