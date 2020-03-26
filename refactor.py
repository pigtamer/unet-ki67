mode = "mac"

import os

if mode == "mac":
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import argparse
from model import *
from data import *
from utils import *
from color_proc import *
import time


class SegModel():
    def __init__(self,
                 input_size, output_size,
                 model_name,
                 loss_name,
                 data_name,
                 batch_size=32, batch_size_val=32,
                 learning_rate=1E-3,
                 framework="k",
                 num_epochs=1,
                 checkpoint_period=1,
                 multi_gpu=False,
                 load_model = None
                 ):
        # only the params needed for building segmentation net

        self.multi_gpu = multi_gpu

        self.input_size = input_size
        self.output_size = (output_size, output_size)

        self.model_name = model_name
        self.loss_name = loss_name  # focalja, bce, bceja, ja
        self.data_name = data_name

        self.load_model = load_model

        self.bs = batch_size
        self.bs_v = batch_size_val

        self.lr = learning_rate

        self.checkpoint_period = 5
        self.flag_test, self.flag_continue = 1, 0
        self.continue_step = (0, 0)
        self.num_epoches = 0
        self.framework = ""

    def build_unet(self):
        self.model = unet(input_size=(self.input_size))

        if self.multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.model = multi_gpu_model(self.model, gpus=2)
                self.model.compile(optimizer=Adam(lr=self.lr),
                                   loss=sm.losses.binary_focal_jaccard_loss,
                                   metrics=[sm.metrics.iou_score, 'accuracy'])
        else:
            self.model.compile(optimizer=Adam(lr=self.lr),
                               loss=loss_dict[self.loss_name],
                               metrics=[sm.metrics.iou_score, 'accuracy'])

    def train(self,
              batch_size=32, batch_size_val=32,
              learning_rate=1E-3,
              framework="k",
              num_epochs=1,
              checkpoint_period=1,
              ):
        self.step_num = 16000 // self.bs
        self.lrstr = "{:.2e}".format(self.lr)

        pass

    def test(self):
        self.test_size = (0, 0)

    def report(self):
        print("*-" * 20, "\n",
              "Testing" if self.istest else "Training", " on "
                                                      "MultiGPU" if self.multi_gpu else "Single GPU", "\n",

              "Model Name: %s" % model_path, "\n",
              "Training Data: %s" % train_path, "\n",
              "Validation Data %s" % val_path, "\n",
              "Total epochs to go: %d, " % num_epoches,
              "save model at %s every %d epochs" % (model_dir, checkpoint_period),
              "\n",
              "Learning Rate %f" % lr, "\n",
              "Batch size %d" % bs, "\n",
              "Image size %d" % edge_size, "\n",
              "%d steps per epoch" % step_num, "\n",
              "*-" * 20, "\n",
              )


results = model.predict_generator(testGene, 30, verbose=1)
