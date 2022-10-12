from configs import *
from utils import *

# from kdeeplabv3.model import Deeplabv3
import numpy as np
import os

import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import losses, callbacks
# import tensorflow_io as tfio
from tensorflow.keras.models import *
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import segmentation_models as sm

model_dict = {}
loss_dict = {
    "bceja": sm.losses.bce_jaccard_loss,
    "focal10ja": sm.losses.BinaryFocalLoss(alpha=0.02, gamma=100)
    + sm.losses.jaccard_loss,
    "ja": sm.losses.jaccard_loss,
    "focal": sm.losses.BinaryFocalLoss(alpha=0.1, gamma=2),
    "bce": sm.losses.binary_crossentropy,
    "focalja": sm.losses.binary_focal_jaccard_loss,
    "focaldice": sm.losses.binary_focal_dice_loss,
    "dice": sm.losses.dice_loss,
    "bcedice": sm.losses.bce_dice_loss,
    "l1": losses.mean_absolute_percentage_error,
    "l2": losses.mean_squared_error,
}

#%%
def smunet(loss="focal", pretrained_weights=None):
    gpu_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICES, cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with gpu_strategy.scope():
        model = sm.Unet(
            backbone_name="densenet121",
            input_shape=(256, 256, 3),
            classes=1,
            activation="sigmoid",
            weights=None,
            encoder_weights="imagenet",
            encoder_freeze=False,
            encoder_features="default",
            decoder_block_type="upsampling",
            decoder_filters=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True,
        )
        # model = tf.keras.models.experimental.SharpnessAwareMinimization(
        #     model, rho=0.05, num_batch_splits=None, name=None
        # )
        opt = tf.optimizers.Adam(lr)
        model.compile(
            optimizer=opt,
            loss=loss_dict[loss],
            metrics=[
                sm.metrics.iou_score,
                # CohenKappaImg(num_classes=2, sparse_labels=True),
                "accuracy",
            ],
        )
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


def kumatt(loss="focal", pretrained_weights=None):
    from keras_unet_collection import models as kum

    gpu_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICES, cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with gpu_strategy.scope():
        model = kum.att_unet_2d(
            input_size=(256, 256, 3),
            filter_num=[16, 32, 64, 128, 256],
            n_labels=1,
            stack_num_down=2,
            stack_num_up=2,
            activation="ReLU",
            atten_activation="ReLU",
            attention="add",
            output_activation="Sigmoid",
            batch_norm=True,
            pool=False,
            unpool=True,
            backbone="DenseNet121",
            weights="imagenet",
            freeze_backbone=False,
            freeze_batch_norm=False,
            name="attunet",
        )

        model = kum.unet_2d(
            input_size=(256, 256, 3),
            filter_num=[16, 32, 64, 128, 256],
            n_labels=1,
            stack_num_down=2,
            stack_num_up=2,
            activation="ReLU",
            output_activation="Sigmoid",
            batch_norm=True,
            pool=False,
            unpool=True,
            backbone="DenseNet121",
            weights="imagenet",
            freeze_backbone=False,
            freeze_batch_norm=False,
            name="unet",
        )
        opt = tf.optimizers.Adam(lr)
        model.compile(
            optimizer=opt,
            loss=loss_dict[loss],
            metrics=[
                sm.metrics.iou_score,
                # CohenKappaImg(num_classes=2, sparse_labels=True),
                "accuracy",
            ],
        )
        if pretrained_weights:
            model.load_weights(pretrained_weights)
    return model


def deeplab(
    loss="focal",
    input_size=(256, 256, 3),
    lr=1e-3,
    classes=1,
    os=8,
    pretrained_weights=None,
):
    gpu_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICES, cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with gpu_strategy.scope():
        model = Deeplabv3(
            input_shape=input_size,
            # backbone="xception",
            activation="sigmoid",
            OS=os,
            classes=classes,
        )
        opt = tf.optimizers.Adam(lr)
        model.compile(
            optimizer=opt,
            loss=loss_dict[loss],
            metrics=[
                sm.metrics.iou_score,
                CohenKappaImg(num_classes=2, sparse_labels=True),
                "accuracy",
            ],
        )
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model


#%%
def unet(
    loss="bceja",
    input_size=(256, 256, 3),
    lr=1e-3,
    pretrained_weights=None,
    pl=[64, 128, 256, 512, 1024],
):
    gpu_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICES, cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with gpu_strategy.scope():

        def resblock(layer_input, filters, f_size=3):
            r = Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(
                layer_input
            )
            r = LeakyReLU(alpha=0.2)(r)
            r = BatchNormalization()(r)
            r = Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(r)
            r = LeakyReLU(alpha=0.2)(r)
            r = Add()([r, layer_input])
            return BatchNormalization()(r)

        def resblockn(n, layer_input, filters, f_size=3):
            x = layer_input
            for k in range(n):
                x = resblock(x, filters, f_size)
            return x

        inputs = Input(input_size)
        conv1 = Conv2D(
            pl[0], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(inputs)
        conv1 = Conv2D(
            pl[0], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(
            pl[1], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool1)
        conv2 = Conv2D(
            pl[1], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv2)

        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(
            pl[2], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool2)
        conv3 = Conv2D(
            pl[2], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(
            pl[3], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool3)
        conv4 = Conv2D(
            pl[3], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv4)
        conv4 = BatchNormalization()(conv4)

        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(
            pl[4], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool4)
        conv5 = Conv2D(
            pl[4], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv5)
        conv5 = BatchNormalization()(conv5)

        drop5 = Dropout(0.5)(conv5)

        # res5 = resblockn(9, drop5, 1024)

        up6 = Conv2D(
            pl[3], 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(drop5))

        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(
            pl[3], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge6)
        conv6 = Conv2D(
            pl[3], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(
            pl[2], 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(
            pl[2], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge7)
        conv7 = Conv2D(
            pl[2], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(
            pl[1], 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(
            pl[1], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge8)
        conv8 = Conv2D(
            pl[1], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv8)
        conv8 = BatchNormalization()(conv8)
        up9 = Conv2D(
            pl[0], 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(
            pl[0], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge9)
        conv9 = Conv2D(
            pl[0], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        conv9 = Conv2D(
            2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)
        model = Model(inputs=inputs, outputs=conv10)

        opt = tf.optimizers.Adam(lr)
        model.compile(
            optimizer=opt,
            loss=loss_dict[loss],
            metrics=[
                sm.metrics.iou_score,
                CohenKappaImg(num_classes=2, sparse_labels=True),
                "accuracy",
            ],
        )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# %%
def unethd(
    loss="focal",
    input_size=(256, 256, 3),
    lr=1e-3,
    pretrained_weights=None,
    pl=[64, 128, 256, 512, 1024],
):
    gpu_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICES, cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with gpu_strategy.scope():

        def resblock(layer_input, filters, f_size=3):
            r = Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(
                layer_input
            )
            r = LeakyReLU(alpha=0.2)(r)
            r = BatchNormalization()(r)
            r = Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(r)
            r = LeakyReLU(alpha=0.2)(r)
            r = Add()([r, layer_input])
            return BatchNormalization()(r)

        def resblockn(n, layer_input, filters, f_size=3):
            x = layer_input
            for k in range(n):
                x = resblock(x, filters, f_size)
            return x

        inputs = Input(input_size)
        conv1 = Conv2D(
            pl[0], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(inputs)
        conv1 = Conv2D(
            pl[0], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(
            pl[1], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool1)
        conv2 = Conv2D(
            pl[1], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv2)

        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(
            pl[2], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool2)
        conv3 = Conv2D(
            pl[2], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(
            pl[3], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool3)
        conv4 = Conv2D(
            pl[3], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv4)
        conv4 = BatchNormalization()(conv4)

        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(
            pl[4], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool4)
        conv5 = Conv2D(
            pl[4], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv5)
        conv5 = BatchNormalization()(conv5)

        drop5 = Dropout(0.5)(conv5)

        # res5 = resblockn(9, drop5, 1024)

        up6 = Conv2D(
            pl[3], 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(drop5))

        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(
            pl[3], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge6)
        conv6 = Conv2D(
            pl[3], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(
            pl[2], 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(
            pl[2], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge7)
        conv7 = Conv2D(
            pl[2], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(
            pl[1], 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(
            pl[1], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge8)
        conv8 = Conv2D(
            pl[1], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv8)
        conv8 = BatchNormalization()(conv8)
        up9 = Conv2D(
            pl[0], 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(
            pl[0], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge9)
        conv9 = Conv2D(
            pl[0], 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        conv9 = Conv2D(
            2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(3, 1, activation="sigmoid")(conv9)
        model = Model(inputs=inputs, outputs=conv10)

        opt = tf.optimizers.Adam(lr)
        model.compile(
            optimizer=opt,
            loss=loss_dict[loss],
            metrics=[
                sm.metrics.iou_score,
                CohenKappaImg(num_classes=2, sparse_labels=True),
                "accuracy",
            ],
        )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


from classification_models.tfkeras import Classifiers

n_classes = 2
# build model
from tensorflow.keras.applications.resnet50 import ResNet50


def u_res50enc(
    backbone=None,
    pretrained_weights=None,
    input_size=(256, 256, 3),
    lr=1e-3,
    multi_gpu=False,
    loss="l1",
):

    strategy = tf.distribute.MirroredStrategy(
        devices=DEVICES, cross_device_ops=tf.distribute.NcclAllReduce()
    )
    with strategy.scope():
        #         inputs = Input(input_size)
        Net, preprocess_input = Classifiers.get("resnet50")
        base_model = Net(input_shape=(256, 256, 3), weights=None, include_top=False)

        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        beforeGAP = keras.layers.Flatten()(base_model.output)

        output = keras.layers.Dense(n_classes, activation="sigmoid")(x)
        backbone = keras.models.Model(
            inputs=[base_model.input], outputs=[output, x, beforeGAP]
        )

        # backbone.load_weights("/home/cunyuan/code/nuc64cls/modelol50.h5")
        backbone = backbone

        def resblock(layer_input, filters, f_size=3):
            r = Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(
                layer_input
            )
            r = LeakyReLU(alpha=0.2)(r)
            r = BatchNormalization()(r)
            r = Conv2D(filters, kernel_size=f_size, strides=1, padding="same")(r)
            r = LeakyReLU(alpha=0.2)(r)
            r = Add()([r, layer_input])
            return BatchNormalization()(r)

        def resblockn(n, layer_input, filters, f_size=3):
            x = layer_input
            for k in range(n):
                x = resblock(x, filters, f_size)
            return x

        #         inputs = Input(input_size)
        #         inputl = backbone.get_layer("data")
        skip0 = backbone.get_layer("bn_data").output
        skip1 = backbone.get_layer("relu0").output
        skip2 = backbone.get_layer("add_1").output

        skip3 = backbone.get_layer("add_6").output
        skip4 = backbone.get_layer("add_11").output

        #         res5 = resblockn(9, skip4, 2048)
        res5 = backbone.get_layer("add_15").output
        #         res5 = MaxPool2D()(res5)

        up6 = Conv2D(
            512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(res5))
        merge6 = concatenate([skip4, up6], axis=3)

        conv6 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge6)
        conv6 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(
            256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([skip3, up7], axis=3)

        conv7 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge7)
        conv7 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(
            128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([skip2, up8], axis=3)
        conv8 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge8)
        conv8 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv8)
        conv8 = BatchNormalization()(conv8)
        up9 = Conv2D(
            64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([skip1, up9], axis=3)
        conv9 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge9)
        conv9 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        conv9 = Conv2D(
            2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        conv9 = BatchNormalization()(conv9)

        up91 = Conv2D(
            64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv9))
        merge91 = concatenate([skip0, up91], axis=3)
        conv91 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge91)
        conv91 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv91)
        conv91 = Conv2D(
            2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv91)
        conv91 = BatchNormalization()(conv91)

        conv10 = Conv2D(1, 1, activation="sigmoid")(conv91)
        model = Model(inputs=backbone.input, outputs=conv10)
        # model = multi_gpu_model(model, gpus=2)
    model.compile(
        keras.optimizers.Adam(1e-3, 0.9),
        loss=loss_dict[loss],
        metrics=[
            sm.metrics.iou_score,
            # CohenKappaImg(num_classes=2, sparse_labels=True),
            "accuracy",
        ],
    )

    # model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model