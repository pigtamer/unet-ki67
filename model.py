# from configs import *
# from utils import *

# from kdeeplabv3.model import Deeplabv3
import numpy as np
import os

import skimage.io as io
import skimage.transform as trans
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


import segmentation_models_pytorch as sm

from utils import DiceLoss

model_dict = {}
# TODO: implement complex losses
loss_dict = {
    "bceja": nn.BCELoss,
    "l1": nn.L1Loss,
    "l2": nn.MSELoss,
}

#%%
def smunet(loss="bceja", pretrained_weights=None):
    model = sm.Unet(
        encoder_name="densenet121",
        classes=1,
        activation="sigmoid",
        encoder_weights="imagenet",
        decoder_use_batchnorm=True,
    )
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model

# =========== test
m = smunet()
print(m)