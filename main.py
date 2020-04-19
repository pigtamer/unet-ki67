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
    val_path = "/Users/cunyuan/DATA/chipwise/val/"
    test_path = "/Users/cunyuan/DATA/test_1024/crop/"
    index_path = "/Users/cunyuan/DATA/ji1024_orig/3e/val1024/"
    # index_path = "/Users/cunyuan/code/tti/cyclegan-ki67/datasets/comparison/v1/"
lr = 1E-3
lrstr = "{:.2e}".format(lr)
edge_size = 256
target_size = (edge_size, edge_size)

test_size = (1024 // (256 // edge_size), 1024 // (256 // edge_size))
# test_size = (256, 256)
bs = 16
bs_v = 4
bs_i = 1
step_num = 14480 // bs

checkpoint_period = 10
flag_test, flag_continue = 1, 0
flag_multi_gpu = 0
continue_step = (0, 0)
num_epoches = 100
framework = "k"
model_name = "unet"
loss_name = "bceja"  # focalja, bce, bceja, ja
data_name = "chipwise"

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
                         train_path=val_path,
                         image_folder='chips',
                         mask_folder='masks',
                         aug_dict={},
                         save_to_dir=None,
                         image_color_mode="rgb",
                         mask_color_mode="grayscale",
                         target_size=target_size)
testGene = testGenerator(test_path, as_gray=False,
                         target_size=target_size)

if mode == "mac":
    indexGene = indexTestGenerator(bs_i,
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
                 multi_gpu=flag_multi_gpu,
                 loss=loss_name)
    # model = unetxx(pretrained_weights=continue_path,
    #                lr=lr)
else:
    model = unet(pretrained_weights=None,
                 input_size=(target_size[0], target_size[1], 3),
                 lr=lr,
                 multi_gpu=flag_multi_gpu,
                 loss=loss_name)
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

val_iters = 1280 // bs_v
# grid search
for k in range(20, 100):
    # continue each model checkpoint
    start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.hdf5" % \
                 (framework, model_name, data_name, loss_name, edge_size, lrstr, continue_step[0] + continue_step[1],
                  k * checkpoint_period)
    model = unet(pretrained_weights=start_path,
                 input_size=(target_size[0], target_size[1], 3),
                 lr=lr,
                 multi_gpu=flag_multi_gpu)
    # model = denseunet(start_path)
    # model = unetxx(start_path,
    #                lr=lr)
    # for k_val, (x, y) in zip(range(val_iters), valGene):
    #     f = model.predict(x, batch_size=bs_v)
    #     # plt.show()
    #     # # print(confusion_matrix(y.reshape(-1,)>0, f.reshape(-1,)>thresh))
    #
    #     if k_val == 0:
    #         f1_max = 0
    #         X, Y, F = x, y, f
    #         thresh_argmax_f1 = 0
    #         print(start_path)
    #         print("Model @ epoch %d" % (k * checkpoint_period), "\n", "-*-" * 10)
    #         for thresh in np.linspace(0, 0.6, 50):
    #             f1 = f1_score(y.reshape(-1, ) > 0, f.reshape(-1, ) > thresh)
    #             if f1 > f1_max:
    #                 f1_max = f1
    #                 thresh_argmax_f1 = thresh
    #         print("Max F1=: ", f1_max, " @ thr: ", thresh_argmax_f1)
    #     else:
    #         Y, F = np.concatenate([Y, y], axis=0), np.concatenate([F, f], axis=0)
    #     print(k_val)
    #
    # iou = jaccard_score(Y.reshape(-1, ) > 0, F.reshape(-1, ) > thresh_argmax_f1)
    # print("IOU= ", iou)
    # print(classification_report(Y.reshape(-1, ) > 0, F.reshape(-1, ) > thresh_argmax_f1))
    #
    # from sklearn.metrics import roc_curve, auc
    #
    # # roc(Y, F, thresh=0)
    # fpr, tpr, _ = roc_curve(Y.ravel(), F.ravel())
    # area_under_curve = auc(fpr, tpr)
    # plt.figure(figsize=(8, 8))
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(area_under_curve))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.grid()
    # plt.show()
    #
    # for kk in range(min(bs_v, 10)):
    #     fig = plt.figure(figsize=(20, 20))
    #     # plt.subplots(2,2)
    #     plt.subplot(221)
    #     plt.imshow(x[kk, :, :, :])
    #     plt.title('Input')
    #     plt.subplot(222)
    #     plt.imshow(y[kk, :, :, 0], cmap='gray')
    #     plt.title('GT')
    #     plt.subplot(223)
    #     plt.imshow(f[kk, :, :, 0], cmap='gray')
    #     plt.title('Pred')
    #     plt.subplot(224)
    #     plt.imshow((f[kk, :, :, 0] > thresh_argmax_f1), cmap='gray')
    #     plt.title('Pred thresh')
    #     fig.tight_layout()
    #     plt.show()

    if mode == "mac":
        num_tp_, num_tn_, num_pred_, num_npred_, num_positive_, num_negative_ = 0, 0, 0, 0, 0, 0
        for kk, (tx, ty, tn) in zip(range(7), indexGene):
            # tx, ty, tn = indexGene.__next__()
            num_tp, num_tn, num_pred, num_npred, num_positive, num_negative = single_prediction(tx, ty, tn, model, 256)
            num_tp_ += num_tp
            num_tn_ += num_tn
            num_pred_ += num_pred
            num_npred_ += num_npred
            num_positive_ += num_positive
            num_negative_ += num_negative
        num_all_ = num_positive_ + num_negative_
        print("F.Prec. %3.2f F.Reca. %3.2f \nT.Prec. %3.2f T.Reca. %3.2f\nAcc. %3.2f"
              % (num_tn_ / num_npred_, num_tn_ / num_negative_, num_tp_ / num_pred_, num_tp_ / num_positive_,
                 (num_tn_ + num_tp_) / (num_negative_ + num_positive_)))
        print("Labelling index Pred. %3.2f\nTrue. %3.2f" % (num_pred_ / num_all_, num_positive_ / num_all_))
    results = model.predict_generator(testGene, 30, verbose=1)
