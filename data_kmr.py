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

from data import adjustData
from model import *
#%%
# mode = "-mac"  # TODO: use argparse instead!!

# if mode == "mac":
#     # use plaidml backend for Mac OS
#     os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# import keras
# import keras.backend as K
# from keras.preprocessing.image import ImageDataGenerator


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


def folds(l_wsis=None, k=5):
    """folds [summary]

    [extended_summary]

    Args:
        l_wsis ([list], optional): [description]. Defaults to None.
        k (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]

    l_wsis = [
        "01_15-1052_Ki67_HE",   #   1   
        "01_14-7015_Ki67_HE",      
        "01_14-3768_Ki67_HE",   
        "01_17-5256_Ki67_HE",   #   2
        "01_17-6747_Ki67_HE",
        "01_17-8107_Ki67_HE",
        "01_15-2502_Ki67_HE",   #   3
        "01_17-7885_Ki67_HE",
        "01_17-7930_Ki67_HE",
    ] """

    def create_divides(l, k):
        if len(l) % k == 0:
            n = len(l) // k
        else:
            n = len(l) // k + 1
        res = [l[i * n : i * n + n] for i in range(k)]
        if res[-1] == []:
            n -= 1
        
        return [l[i * n : i * n + n] for i in range(k)]

    return [([x for x in l_wsis if x not in f], f) for f in create_divides(l_wsis, k)]


def kmrGenerator(
    dataset_path,
    batch_size=4,
    image_folder=None,
    mask_folder=None,
    aug_dict=None,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    flag_multi_class=False,
    num_class=2,
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
) -> tuple:
    """kmrGenerator: Custom generator providing HE-DAB or HE-Mask pair for fit_generator

    [extended_summary]

    Args:
        dataset_path ([string]): path of the dataset
        batch_size (int, optional): batch size. Defaults to 4.
        image_folder (list of strings, optional): folder for he stains. Defaults to None.
        mask_folder (list of strings, optional): folder of masks or DABs. Defaults to None.
        aug_dict (ditionary, optional): data augmentation arguments. Defaults to None.
        image_color_mode (str, optional): color mode of the he images. Defaults to "rgb".
        mask_color_mode (str, optional): color mode of masks/DAB. Defaults to "grayscale".
        image_save_prefix (str, optional): [description]. Defaults to "image".
        mask_save_prefix (str, optional): [description]. Defaults to "mask".
        flag_multi_class (bool, optional): [description]. Defaults to False.
        num_class (int, optional): number of classes. Defaults to 2.
        save_to_dir ([type], optional): [description]. Defaults to None.
        target_size (tuple, optional): 2-D size of training images. Defaults to (256, 256).
        seed (int, optional): seed for random shuffling. Defaults to 1.

    Returns:
        tuple: yield pairs of HE image - Mask or DAB
    """
    he_datagen = ImageDataGenerator(**aug_dict)
    dab_datagen = ImageDataGenerator(**aug_dict)

    he_generator = he_datagen.flow_from_directory(
        dataset_path + "HE/",
        classes=image_folder,
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )

    dab_generator = dab_datagen.flow_from_directory(
        dataset_path + "Mask/",
        classes=image_folder,
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    train_generator = zip(he_generator, dab_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


#%%
# * 2. Cross validation

# %%
# * 3. Tf.data as input pipeline
def load_kmr_tfdata(dataset_path,
                    batch_size=16,
                    wsi_ids=None,
                    stains=["HE", "Mask"],
                    aug=False,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    flag_multi_class=False,
                    num_class=2,
                    save_to_dir=None,
                    target_size=(256, 256),
                    seed=1,
                    cache = None,
                    shuffle_buffer_size=128
                    ) -> tuple:
    def parse_image(file_path):
            img = tf.io.read_file(file_path)
            # convert the compressed string to a 3D uint8 tensor
            img = tfio.experimental.image.decode_tiff(img)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [target_size[0], target_size[1]])
            # resize the image to the desired size.
            return img

    def prepare_for_training(ds, cache=cache, 
                            shuffle_buffer_size=shuffle_buffer_size, batch_size=batch_size):
        # If a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets
        # that don't fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        # ds = ds.shard(num_shards = hvd.size(),
        #                 index = hvd.rank())
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed, 
                        reshuffle_each_iteration=False)
        # Repeat forever
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background
        # while the model is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    data_generator = {}
    for staintype in stains:
        dir_pattern = [dataset_path + "/" + staintype + "/" +  wsi +  
                    "*/Tiles/Tumor/*/*" for wsi in wsi_ids ]
        list_ds = tf.data.Dataset.list_files(dir_pattern, shuffle=True, seed=seed)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # Set `num_parallel_calls` so that multiple images are
        # processed in parallel
        labeled_ds = list_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
        # cache = True, False, './file_name'
        # If the dataset doesn't fit in memory use a cache file,
        # eg. cache='./data.tfcache'
        data_generator[staintype] = prepare_for_training(
            labeled_ds, cache = (cache + '_%s_%d.tfcache'%(staintype, 1E10*np.random.rand()))
            if isinstance(cache, str) else cache)
    train_generator = zip(data_generator["HE"],data_generator["Mask"])
    if aug:
        def data_augmentation(seed):
            return tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='constant', seed=seed),
            layers.experimental.preprocessing.RandomTranslation(
                0.1, 0.1, fill_mode='constant', seed=seed),
            ])
        augimg = data_augmentation(seed)
        augmask = data_augmentation(seed)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        if aug:
            img = augimg(img)
            mask = augmask(mask)
        yield (img, mask)

# train_gen = load_kmr_tfdata(
#         "/home/cunyuan/4tb/Kimura/DATA/TILES_(256, 256)_0.41",
#         ["01_15-1052_Ki67",   #   1   
#         "01_14-7015_Ki67",      
#         "01_14-3768_Ki67",   
#         "01_17-5256_Ki67",   #   2
#         "01_17-6747_Ki67",
#         "01_17-8107_Ki67",],
#         ["IHC",
#         "Mask"])

# for img_pairs in train_gen:
#     plt.figure(); plt.subplot(121);plt.imshow(img_pairs[1][0,:,:,:4])
#     plt.subplot(122);plt.imshow(img_pairs[0][0,:,:,:4]);plt.show()
