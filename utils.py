# from keras import backend as K
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
# import cv2 as cv

# focal loss


def viewim(im, cmap="gray"):
    plt.imshow(im, cmap=cmap)
    plt.show()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-7
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score


def e2e_train(trainGene, valGene, model):
    """
    Function to perform end-to-end training with ki67 ground truth as reference image

    Parameters
    ----------
    trainGene: training image generator
    valGene: validation image generator
    model: model to train.

    Returns: model with updated parameters
    -------

    """
    pass


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
    ]"""

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