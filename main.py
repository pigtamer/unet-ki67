mode = "mac"

import os

if mode == "mac":
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import argparse
from model import *
from data import *
from utils import *
from color_proc import *
import segmentation_models as sm
from segmodel import denseunet, unetxx
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoches", dest="num_epoches",
                    help="int: trainig epoches",
                    type=int, default=50)
parser.add_argument("-bs", "--batch_size", dest="batch_size",
                    help="int: batch size for training",
                    type=int, default=2)
parser.add_argument("-is", "--imsize", dest="input_size",
                    help="int: input size",
                    type=int, default=1024)

args = parser.parse_args()

data_gen_args = dict(rotation_range=5,
                     channel_shift_range=0,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

# On server. full annotated data 16040
train_path = "/home/cunyuan/DATA/ORIG/chipwise/train/"
val_path = "/home/cunyuan/DATA/ORIG/chipwise/val/"
test_path = "/home/cunyuan/DATA/test_1024/k/"
model_dir = "/home/cunyuan/models/"

if mode == "mac":
    model_dir = "/Users/cunyuan/models/"
    train_path = "/Users/cunyuan/DATA/chipwise/train/"
    val_path = "/Users/cunyuan/DATA/chipwise/val1/"
    test_path = "/Users/cunyuan/DATA/test_1024/crop/"
    index_path = "/Users/cunyuan/DATA/ji1024_orig/4d/val1024/"
    index_path = "/Users/cunyuan/code/me/gan/keras-gan/cyclegan/datasets/minimal/"

lr = 1E-3
lrstr = "{:.2e}".format(lr)
edge_size = 128
target_size = (edge_size, edge_size)

test_size = (1024//(256//edge_size), 1024//(256//edge_size))
test_size = (256,256)

bs = 32*4
bs_v = 1
step_num = 15000 // bs

checkpoint_period = 10
flag_test, flag_continue = 1, 0
flag_multi_gpu = 0
continue_step = (0, 0)
num_epoches = 500
framework = "k"
model_name = "unet"
loss_name = "l1"  # focalja, bce, bceja, ja
data_name = "chipwise"

trainGene = trainGenerator(bs,
                           train_path=train_path,
                           image_folder='chips',
                           mask_folder='dab',
                           aug_dict=data_gen_args,
                           save_to_dir=None,
                           image_color_mode="rgb",
                           mask_color_mode="rgb",
                           target_size=target_size)
valGene = trainGenerator(bs_v,
                         train_path=val_path,
                         image_folder='chips',
                         mask_folder='dab',
                         aug_dict={},
                         save_to_dir=None,
                         image_color_mode="rgb",
                         mask_color_mode="rgb",
                         target_size=target_size)
testGene = testGenerator(test_path, as_gray=False,
                         target_size=target_size)

if mode == "mac":
    indexGene = indexTestGenerator(bs_v,
                                   train_path=index_path,
                                   image_folder='chips',
                                   mask_folder='masks',
                                   nuclei_folder="nuclei",
                                   aug_dict={},
                                   save_to_dir=None,
                                   image_color_mode="rgb",
                                   mask_color_mode="grayscale",
                                   target_size=test_size)


model_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+{epoch:02d}.hdf5" % \
             (framework, model_name, data_name, loss_name, edge_size, lrstr, continue_step[1] + continue_step[0])

continue_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.hdf5" % \
                (framework, model_name, data_name, loss_name, edge_size, lrstr, continue_step[0], continue_step[1])

if flag_continue:
    model = unet(pretrained_weights=continue_path,
                 input_size=(target_size[0], target_size[1], 3),
                 lr=lr,
                 loss=loss_name,
                 multi_gpu=flag_multi_gpu)
    # model = unetxx(pretrained_weights=continue_path,
    #                lr=lr)
else:
    model = unet(pretrained_weights=None,
                 input_size=(target_size[0], target_size[1], 3),
                 lr=lr,
                 loss=loss_name,
                 multi_gpu=flag_multi_gpu)
    # model = unetxx(lr=lr)

plot_model(model, to_file="model.svg")
"""
Train the model

Training process summary

"""

print("*-" * 20, "\n",
      "Testing" if flag_test else "Training", " on "
                                              "MultiGPU" if flag_multi_gpu else "Single GPU", "\n",
      "New model" if not flag_continue else "Continue from epoch %d" % (continue_step[1] + continue_step[0]), "\n",
      "Model Name: %s" % model_path, "\n",
      "Training Data: %s" % train_path, "\n",
      "Validation Data %s" % val_path, "\n",
      "Total epochs to go: %d, " % num_epoches, "save model at %s every %d epochs" % (model_dir, checkpoint_period),
      "\n",
      "Learning Rate %f" % lr, "\n",
      "Batch size %d" % bs, "\n",
      "Image size %d" % edge_size, "\n",
      "%d steps per epoch" % step_num, "\n",
      "*-" * 20, "\n",
      )

if not flag_test:
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=checkpoint_period)

    start = time.time()

    model.fit_generator(trainGene,
                        validation_data=valGene,
                        validation_steps=10,
                        steps_per_epoch=step_num,
                        epochs=num_epoches,
                        callbacks=[model_checkpoint])

    print(time.time() - start)

# grid search
for k in range(40, 100):
    # continue each model checkpoint
    start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.hdf5" % \
                 (framework, model_name, data_name, loss_name, edge_size, lrstr, continue_step[0] + continue_step[1],
                  k * checkpoint_period)
    model = unet(pretrained_weights=start_path,
                 input_size=(target_size[0], target_size[1], 3),
                 lr=lr,
                 loss=loss_name,
                 multi_gpu=flag_multi_gpu)
    # model = denseunet(start_path)
    # model = unetxx(start_path,
    #                lr=lr)

    (x, y) = valGene.__next__()
    f = model.predict(x, batch_size=bs_v)
    if mode == "mac":
        while True:
            tx, ty, tn = indexGene.__next__()
            ft = single_prediction(tx, ty, tn, model, edge_size)

    fig = plt.figure(figsize=(20, 20))
    # plt.subplots(2,2)
    plt.subplot(221)
    plt.imshow(x[1, :, :, :])
    plt.title('Input')
    plt.subplot(222)
    plt.imshow(y[1, :, :, :]);
    plt.title('GT')
    plt.subplot(223)
    plt.imshow(f[1, :, :, :]);
    plt.title('Pred')
    fig.tight_layout()
    plt.show()

results = model.predict_generator(testGene, 30, verbose=1)
