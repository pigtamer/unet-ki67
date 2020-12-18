import segmentation_models as sm
from UNetPlusPlus.segmentation_models import Xnet

# from keras.optimizers import *
from utils import jaccard_distance_loss

# define model


def denseunet(pretrained_weights=None, lr=1e-4):
    model = sm.Unet("densenet121", encoder_weights="imagenet")

    model.compile(
        optimizer="adam",
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


def unetxx(pretrained_weights=None, lr=1e-4):
    model = Xnet(
        backbone_name="resnet50", encoder_weights=None, decoder_block_type="transpose"
    )
    model.compile(
        optimizer="adam",
        loss=jaccard_distance_loss,
        metrics=[sm.metrics.iou_score, "accuracy"],
    )
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model
