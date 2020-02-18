import os

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import argparse
from model import *
from data import *
from utils import *
from color_proc import *
import segmentation_models as sm
from segmodel import denseunet
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", dest="load",
                    help="bool: load model to directly infer rather than training",
                    type=int, default=1)
parser.add_argument("-b", "--base", dest="base",
                    help="bool: using additional base network",
                    type=int, default=0)
parser.add_argument("-e", "--epoches", dest="num_epoches",
                    help="int: trainig epoches",
                    type=int, default=50)
parser.add_argument("-bs", "--batch_size", dest="batch_size",
                    help="int: batch size for training",
                    type=int, default=2)
parser.add_argument("-is", "--imsize", dest="input_size",
                    help="int: input size",
                    type=int, default=1024)

parser.add_argument("-lr", "--learning_rate", dest="learning_rate",
                    help="float: learning rate of optimization process",
                    type=float, default=0.0001)
parser.add_argument("-opt", "--optimize", dest="optimize_method",
                    help="optimization method",
                    type=str, default="sgd")

parser.add_argument("-dp", "--data_path", dest="data_path",
                    help="str: the path to dataset",
                    type=str, default="../data/uav/usc/1479/raw/")
# ../../data/uav/usc/1479/output/cropped/
parser.add_argument("-mp", "--model_path", dest="model_path",
                    help="str: the path to load and save model",
                    type=str, default="../params/autofocus/")
parser.add_argument("-tp", "--test_path", dest="test_path",
                    help="str: the path to your test img",
                    type=str, default="../data/dji1.mp4")
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
train_path = "/home/cunyuan/DATA/ORIG/1w6_div/train/"
val_path = "/home/cunyuan/DATA/ORIG/1w6_div/val/"

# # On mac
# train_path = "/Users/cunyuan/DATA/ki67/Ji/orig/"
# val_path = "/Users/cunyuan/DATA/ki67/Ji/orig/"

val_path_broken = "/home/cunyuan/DATA/1d_1024/val/"

test_path = "/home/cunyuan/DATA/test_1024/val/"

lr = 1E-3
edge_size = 128
target_size = (edge_size, edge_size);
bs = 64
bs_v = 100
step_num = 16000 // bs

trainGene = trainGenerator(bs,
                           train_path=train_path,
                           image_folder='chips',
                           mask_folder='masks',
                           aug_dict=data_gen_args,
                           save_to_dir=None,
                           image_color_mode="rgb",
                           mask_color_mode="grayscale",
                           target_size=target_size)
valGene = trainGenerator(bs_v,
                         train_path=test_path,
                         image_folder='chips',
                         mask_folder='masks',
                         aug_dict=data_gen_args,
                         save_to_dir=None,
                         image_color_mode="rgb",
                         mask_color_mode="grayscale",
                         target_size=target_size)
testGene = testGenerator(test_path, as_gray=False,
                         target_size=target_size)
model_dir = "/home/cunyuan/models/"

checkpoint_period = 5
flag_test, flag_continue = 1, 1
flag_multi_gpu = 0
continue_step = (0, 20)

"""
Training process summary
"""

summary_string = ""

model_path = model_dir + 'k-unet__1w6div_bceja_%d_lr-3_ep%02d+{epoch:02d}.hdf5' % (edge_size, continue_step[1])

continue_path = model_dir + 'k-unet__1w6div_bceja_%d_lr-3_ep%02d+%02d.hdf5' % \
                (edge_size, continue_step[0], continue_step[1])

if flag_continue:
    model = unet(pretrained_weights=continue_path,
                 input_size=(target_size[0], target_size[1], 3),
                 lr=lr,
                 multi_gpu=flag_multi_gpu)
else:
    model = unet(pretrained_weights=None,
                 input_size=(target_size[0], target_size[1], 3),
                 lr=lr,
                 multi_gpu=flag_multi_gpu)
    # model = denseunet()

plot_model(model, to_file="model.svg")
"""

Train the model

"""

if not flag_test:
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=checkpoint_period)

    start = time.time()

    model.fit_generator(trainGene,
                        validation_data=valGene,
                        validation_steps=10,
                        steps_per_epoch=step_num,
                        epochs=20,
                        callbacks=[model_checkpoint])

    print(time.time() - start)

# grid search
for k in range(1, 30):
    # continue each model checkpoint
    start_path = model_dir + 'k-unet__1w6div_bceja_%d_lr-3_ep%02d+%02d.hdf5' % \
                 (edge_size, continue_step[1], k * checkpoint_period)
    model = unet(pretrained_weights=start_path,
                 input_size=(target_size[0], target_size[1], 3),
                 lr=lr,
                 multi_gpu=flag_multi_gpu)

    (x, y) = valGene.__next__()
    f = model.predict(x, batch_size=bs_v)

    # # print(confusion_matrix(y.reshape(-1,)>0, f.reshape(-1,)>thresh))
    f1_max = 0;
    thresh_argmax_f1 = 0;
    print(start_path)
    print("Model @ epoch %d" % (k * checkpoint_period), "\n", "-*-" * 10)
    for thresh in np.linspace(0, 0.6, 50):
        f1 = f1_score(y.reshape(-1, ) > 0, f.reshape(-1, ) > thresh)
        if f1 > f1_max:
            f1_max = f1
            thresh_argmax_f1 = thresh

    iou = jaccard_score(y.reshape(-1, ) > 0, f.reshape(-1, ) > thresh_argmax_f1)
    print("IOU= ", iou)
    print("Max F1=: ", f1_max, " @ thr: ", thresh_argmax_f1)
    print(classification_report(y.reshape(-1, ) > 0, f.reshape(-1, ) > thresh_argmax_f1))

    roc(y, f, thresh=0)
    fig = plt.figure(figsize=(20, 20))
    # plt.subplots(2,2)
    plt.subplot(221)
    plt.imshow(x[1, :, :, :])
    plt.title('Input')
    plt.subplot(222)
    plt.imshow(y[1, :, :, 0], cmap='gray');
    plt.title('GT')
    plt.subplot(223)
    plt.imshow(f[1, :, :, 0], cmap='gray');
    plt.title('Pred')
    plt.subplot(224)
    plt.imshow((f[1, :, :, 0] > thresh_argmax_f1), cmap='gray')
    plt.title('Pred thresh')
    fig.tight_layout()
    plt.show()

results = model.predict_generator(testGene, 30, verbose=1)
# saveResult("/tmp/pycharm_project_947/data/ki67/", results)
