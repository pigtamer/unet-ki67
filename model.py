mode = "tbm"

if mode == "mac":
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    
import numpy as np
import os

import skimage.io as io
import skimage.transform as trans
import numpy as np

if mode!="mac":
    import tensorflow as tf
    from tensorflow import keras
    import horovod.tensorflow.keras as hvd
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import multi_gpu_model
from utils import *
from tensorflow.keras import losses, callbacks
import tensorflow_io as tfio
from tensorflow.keras.models import *
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import segmentation_models as sm



# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
opt = tf.optimizers.Adam(0.001 * hvd.size())

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

#from segmodel import *

loss_dict = {
    "bceja": sm.losses.bce_jaccard_loss,
    "ja": sm.losses.jaccard_loss,
    "focal": sm.losses.binary_focal_loss,
    "bce":sm.losses.binary_crossentropy,
    "focalja": sm.losses.binary_focal_jaccard_loss,
    "focaldice":sm.losses.binary_focal_dice_loss,
    "dice": sm.losses.dice_loss,
    "bcedice": sm.losses.bce_dice_loss,
    "l1": losses.mean_absolute_percentage_error,
    "l2": losses.mean_squared_error
}

def unet(pretrained_weights=None, input_size=(256, 256, 3), lr=1E-3, multi_gpu=False, loss="l1"):
    # 所有等号左侧其实不是层而是张量吗...
    # 是的！ 因为这里使用了keras的函数式API。每一个层都是可以调用的...而在左边返回输出的张量
    def resblock(layer_input, filters, f_size=3):
        r = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
        r = LeakyReLU(alpha=0.2)(r)
        r = BatchNormalization()(r)
        r = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(r)
        r = LeakyReLU(alpha=0.2)(r)
        r = Add()([r, layer_input])
        return BatchNormalization()(r)

    def resblockn(n, layer_input, filters, f_size=3):
        x = layer_input
        for k in range(n):
            x = resblock(x, filters, f_size)
        return x

    if multi_gpu:
        # strategy = tf.distribute.MirroredStrategy()
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        # with strategy.scope():
            # 要在scope之内完成模型构建编译
            # TODO: 重写本段
            # pass
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv2)

        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)

        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)

        drop5 = Dropout(0.5)(conv5)

        # res5 = resblockn(9, drop5, 1024)


        up6 = Conv2D(512, 2, activation='relu', padding='same',
                    kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))

        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same',
                    kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same',
                    kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same',
                    kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        model = Model(inputs=inputs, outputs=conv10)
        #model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer=opt,
                        loss=loss_dict[loss], metrics=[sm.metrics.iou_score, 'accuracy'])
    else:
        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=opt,
                      loss=loss_dict[loss], metrics=[sm.metrics.iou_score, 'accuracy'])

    # model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def smunet(loss="focal",
            pretrained_weights = None):
    model = sm.Unet(backbone_name = 'densenet121',
                    input_shape=(None, None, 3),
                    classes=1,
                    activation='sigmoid',
                    weights=None,
                    encoder_weights='imagenet',
                    encoder_freeze=False,
                    encoder_features='default',
                    decoder_block_type='upsampling',
                    decoder_filters=(256, 128, 64, 32, 16),
                    decoder_use_batchnorm=True)
    model.compile(optimizer=opt,
                loss=loss_dict[loss], metrics=[sm.metrics.iou_score, 'accuracy'])
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model

model_dict = {
    "unet": unet,
    # "unetxx": unetxx,
    # "unet++": unetxx,
    # "denseunet": denseunet
}
