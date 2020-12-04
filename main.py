#%%
# %matplotlib inline
import configs

from mpi4py import MPI
comm = MPI.COMM_WORLD
print(comm.Get_size())

import os,argparse
from model import *
from data import *
from data_kmr import *

from utils import *
from pathlib import Path
from color_proc import *
import segmentation_models as sm
# from segmodel import denseunet, unetxx
import time
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
)

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-e",
#     "--epoches",
#     dest="num_epoches",
#     help="int: trainig epoches",
#     type=int,
#     default=50,
# )
# parser.add_argument(
#     "-bs",
#     "--batch_size",
#     dest="batch_size",
#     help="int: batch size for training",
#     type=int,
#     default=2,
# )
# parser.add_argument(
#     "-is", "--imsize", dest="input_size", help="int: input size", type=int, default=1024
# )

# args = parser.parse_args()

data_gen_args = dict(
    rotation_range=360,
    channel_shift_range=0,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="constant",
)

# On server. full annotated data 16040
HOME_PATH = str(Path.home())


train_path = HOME_PATH + "/4tb/Kimura/DATA/TILES_(256, 256)_0.25/"
val_path = HOME_PATH + "/4tb/Kimura/DATA/TILES_(256, 256)_0.25/"
test_path = HOME_PATH + "/DATA/test_1024/k/"
model_dir = HOME_PATH + "/models/"

if mode == "tbm":
    STG_PATH = "/gs/hs0/tga-yamaguchi.m/ji"
    train_path = STG_PATH + "/TILES_(256, 256)/"
    val_path = STG_PATH + "/TILES_(256, 256)/"
    test_path = HOME_PATH + "/DATA/test_1024/k/"
    model_dir = STG_PATH + "/models/"
if mode == "mac":
    model_dir = "/Users/cunyuan/models/"
    train_path = "/Users/cunyuan/DATA/chipwise/train/"
    val_path = "/Users/cunyuan/DATA/chipwise/val/"
    val_path = "/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/0.36/val/"
    test_path = "/Users/cunyuan/DATA/test_1024/crop/"
    # index_path = "/Users/cunyuan/DATA/ji1024_orig/val1024/"
    # index_path = "/Users/cunyuan/code/tti/cyclegan-ki67/datasets/comparison/v1/"
    # index_path="/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/0.36/val/"
    # index_path = "/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/0.36/results/200/2502/"
    index_path = "/Users/cunyuan/DATA/Kimura/EMca別症例_WSIとLI算出領域/LI算出領域/17-7885/my2048/"

lr = 1e-3
initial_lr = lr*hvd.size()
# initial_lr = lr*hvd.size()
lrstr = "{:.2e}".format(lr)
edge_size = 256
target_size = (edge_size, edge_size)

# test_size = (1024 // (256 // edge_size), 1024 // (256 // edge_size))
# test_size = (3328, 3328)
# test_size = (1536, 1536)
test_size = (2048, 2048)

bs = 32
bs_v = 16
bs_i = 1
# step_num = 33614 // bs # 0.41
# step_num = 108051 // bs # 0.25
# step_num = 33498 // bs
# step_num = 585891 // bs # all tumor
# step_num = 162721 // bs # G1 tumor
step_num = 466272*2 // bs # G123 tumor, 3div
verbose = 1

checkpoint_period = 5
flag_test, flag_continue = 0, 1
flag_multi_gpu = 1
continue_step = (0, 31) # start epoch, total epochs trained
initial_epoch = continue_step[0] + continue_step[1]
num_epoches = 300
framework = "hvd-tfk"
model_name = "dense121-unet"
loss_name = "bceja"  # focalja, bce, bceja, ja, dice...
data_name = "kmr-G1G2G3-9x3-123"

configstring = "%s_%s_%s_%s_%d_ndx%d_lr%s.h5" % (
    framework,
    model_name,
    data_name,
    loss_name,
    edge_size,
    hvd.size(), # number of nodes, ndx
    lrstr
    )

if mode != "mac":
    logdir = "/gs/hs0/tga-yamaguchi.m/ji/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + configstring
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    # ! This function is ineffective if the learning rate is scheduled in horovod callback
    """
    learning_rate = 1E-2
    if epoch > 1:
        learning_rate = 1E-3
    if epoch > 5:
        learning_rate = 1E-3
    if epoch > 50:
        learning_rate = 1E-3
    if epoch > 100:
        learning_rate = 1E-4
    if epoch > 200:
        learning_rate = 1E-4
    
    tf.summary.scalar("learning rate", data=learning_rate, step=epoch)
    return learning_rate

# def custom_dashboard(epoch, step):

lr_callback = LearningRateScheduler(lr_schedule)
# step_callback = keras.callbacks.Callback()
if mode != "mac":
    tensorboard_callback = TensorBoard(log_dir=logdir)


fold = folds(
    l_wsis=[
        k + ""
        for k in [
            "01_14-7015_Ki67", #1 22091
            "01_17-5256_Ki67", #2 54923
            "01_17-7885_Ki67", #3 42635 --> 466272

            "01_15-1052_Ki67", #1 34251
            "01_17-6747_Ki67", #2 66635
            "01_15-2502_Ki67", #3 136715
        
            "01_14-3768_Ki67", #1 106379
            "01_17-8107_Ki67", #2 69097
            "01_17-7930_Ki67", #3 53195
        ]
    ],
    k=3,
)
print(fold[0][0])
print(fold[0][1])
trainGene = load_kmr_tfdata(
    dataset_path = train_path,
    batch_size=bs,
    wsi_ids=fold[0][0],
    aug=False,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    flag_multi_class=False,
    num_class=2,
    save_to_dir=None,
    target_size=target_size,
    # cache='/gs/hs0/tga-yamaguchi.m/ji/train',
    cache=False,
    shuffle_buffer_size=128,
    seed=hvd.rank()
)
valGene = load_kmr_tfdata(
    dataset_path=val_path,
    batch_size=bs_v,
    wsi_ids=fold[0][1],
    aug=False,
    save_to_dir=None,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    target_size=target_size,
    # cache='/gs/hs0/tga-yamaguchi.m/ji/val',
    cache=False,
    shuffle_buffer_size=128,
    seed=hvd.rank()
)
testGene = testGenerator(test_path, as_gray=False, target_size=target_size)

if mode == "mac":
    indexGene = indexTestGenerator(
        bs_i,
        train_path=index_path,
        image_folder="he",
        mask_folder="masks",
        nuclei_folder="ihc",
        aug_dict={},
        save_to_dir=None,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        nuclei_color_mode="rgb",
        target_size=test_size,
    )

model_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+{epoch:02d}.h5" % (
    framework,
    model_name,
    data_name,
    loss_name,
    edge_size,
    lrstr,
    continue_step[1] + continue_step[0],
)

continue_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
    framework,
    model_name,
    data_name,
    loss_name,
    edge_size,
    lrstr,
    continue_step[0],
    continue_step[1],
)

if flag_continue:
    # model = unet(
    #     pretrained_weights=continue_path,
    #     input_size=(target_size[0], target_size[1], 3),
    #     lr=lr,
    #     multi_gpu=flag_multi_gpu,
    #     loss=loss_name,
    # )
    # model = unetxx(pretrained_weights=continue_path,
    #                lr=lr)
    sm.set_framework('tf.keras')
    model = smunet(loss=loss_name,
                    pretrained_weights=continue_path)
else:
    model = unet(
         pretrained_weights=None,
         input_size=(target_size[0], target_size[1], 3),
         lr=lr,
         multi_gpu=flag_multi_gpu,
         loss=loss_name,
    )
    sm.set_framework('tf.keras')
    model = smunet(loss=loss_name)

# plot_model(model, to_file="./model.svg")
"""
Train the model

Training process summary

"""

print(
    "*-" * 20,
    "\n",
    "Testing" if flag_test else "Training",
    " on " "Multi GPU" if flag_multi_gpu else "Single GPU",
    "\n",
    "New model"
    if not flag_continue
    else "Continue from epoch %d" % (continue_step[1] + continue_step[0]),
    "\n",
    "Model Name: %s" % model_path,
    "\n",
    "Training Data: %s" % train_path,
    "\n",
    "Validation Data %s" % val_path,
    "\n",
    "Total epochs to go: %d, " % num_epoches,
    "save model at %s every %d epochs" % (model_dir, checkpoint_period),
    "\n",
    "Learning Rate %f" % lr,
    "\n",
    "Batch size %d" % bs,
    "\n",
    "Image size %d" % edge_size,
    "\n",
    # "%d steps per epoch" % step_num,
    # "\n",
    "*-" * 20,
    "\n",
)
#%%
# for hd, k in zip(trainGene, range(10)):
#     im = hd[0][0]
#     dab = hd[1][0]
#     plt.figure(figsize=(8, 4), dpi=300)
#     plt.tight_layout()
#     plt.subplot(121)
#     plt.imshow(im)
#     plt.axis("off")
#     plt.subplot(122)
#     plt.imshow(dab[:, :, 0])
#     plt.axis("off")
    # plt.show()
#%%
print("============\n"*3,hvd.size(),"\n","============\n"*3,)
if not flag_test:
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor="loss",
        verbose=verbose,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq=checkpoint_period*step_num//hvd.size(),
    )

    start = time.time()
    callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, initial_lr=initial_lr,
                                             verbose=verbose),

    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=5, end_epoch=10, multiplier=1.,
                                               initial_lr=initial_lr),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=10, end_epoch=50, multiplier=1, initial_lr=initial_lr),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=50, end_epoch=200, multiplier=1, initial_lr=initial_lr),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=200, multiplier=1e-1, initial_lr=initial_lr),
    ]

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(model_checkpoint)
        callbacks.append(tensorboard_callback)
        # print(model.summary())
    training_history = model.fit(
        trainGene,
        validation_data=valGene,
        validation_freq=5,
        validation_steps=100, # 0.41:178 0.25:633
        steps_per_epoch=step_num // hvd.size(),
        epochs=num_epoches,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    print(time.time() - start)

val_iters = 1280 // bs_v
# grid search
for k in range(3, 100):
    # continue each model checkpoint
    start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
        framework,
        model_name,
        data_name,
        loss_name,
        edge_size,
        lrstr,
        continue_step[0] + continue_step[1],
        k * checkpoint_period,
    )
    model = unet(
        pretrained_weights=start_path,
        input_size=(target_size[0], target_size[1], 3),
        lr=lr,
        multi_gpu=flag_multi_gpu,
    )
    # model = denseunet(start_path)
    # model = unetxx(start_path,
    #                lr=lr)
    for k_val, (x, y) in zip(tqdm(range(val_iters)), valGene):
        f = model.predict(x, batch_size=bs_v)
        # plt.show()
        # # print(confusion_matrix(y.reshape(-1,)>0, f.reshape(-1,)>thresh))

        if k_val == 0:
            X, Y, F = x, y, f
            thresh_argmax_f1 = 0
            print(start_path)
            print("Model @ epoch %d" % (k * checkpoint_period), "\n", "-*-" * 10)

        else:
            Y, F = np.concatenate([Y, y], axis=0), np.concatenate([F, f], axis=0)

        # print(k_val)

    # f1_max = 0
    # for thresh in np.linspace(0, 1, 5):
    #     f1 = f1_score(Y.reshape(-1, ) > 0, F.reshape(-1, ) > thresh)
    #     if f1 > f1_max:
    #         f1_max = f1
    #         thresh_argmax_f1 = thresh
    #     print(thresh)
    # print("Max F1=: ", f1_max, " @ thr: ", thresh_argmax_f1)

    thresh_argmax_f1 = 0.5
    iou = jaccard_score(Y.reshape(-1,) > 0, F.reshape(-1,) > thresh_argmax_f1)
    print("IOU= ", iou)
    print(classification_report(Y.reshape(-1,) > 0, F.reshape(-1,) > thresh_argmax_f1))

    from sklearn.metrics import roc_curve, auc

    # roc(Y, F, thresh=0)
    fpr, tpr, _ = roc_curve(Y.ravel(), F.ravel())
    area_under_curve = auc(fpr, tpr)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label="AUC = {:.3f}".format(area_under_curve))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()
    plt.show()

    for kk in range(min(bs_v, 10)):
        fig = plt.figure(figsize=(20, 20))
        # plt.subplots(2,2)
        plt.subplot(221)
        plt.imshow(x[kk, :, :, :])
        plt.title("Input")
        plt.subplot(222)
        plt.imshow(y[kk, :, :, 0], cmap="gray")
        plt.title("GT")
        plt.subplot(223)
        plt.imshow(f[kk, :, :, 0], cmap="gray")
        plt.title("Pred")
        plt.subplot(224)
        plt.imshow((f[kk, :, :, 0] > thresh_argmax_f1), cmap="gray")
        plt.title("Pred thresh")
        fig.tight_layout()
        plt.show()

    if mode == "mac":
        num_tp_, num_tn_, num_pred_, num_npred_, num_positive_, num_negative_ = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        avgiou = 0
        li_mask = cv.cvtColor(
            cv.imread(
                "/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/0.36/results/200/2502/limask/01_15-2502_Ki67_HE (1, x=130037, y=69624, w=2056, h=2056)_1_mask.tif"
            ),
            cv.COLOR_BGR2GRAY,
        )
        for kk, (tx, ty, tn) in zip(range(1000), indexGene):
            # tx, ty, tn = indexGene.__next__()
            # if kk< 500: continue
            (
                num_tp,
                num_tn,
                num_pred,
                num_npred,
                num_positive,
                num_negative,
                iou,
                res,
            ) = single_prediction(tx, ty, tn, model, None, 256)
            num_tp_ += num_tp
            num_tn_ += num_tn
            num_pred_ += num_pred
            num_npred_ += num_npred
            num_positive_ += num_positive
            num_negative_ += num_negative
            avgiou += iou
            # plt.show()
            plt.imsave(
                "/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/0.36/results/200/2502/test_seq_%d.png"
                % kk,
                res,
            )
            plt.imsave(
                "/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/0.36/results/200/2502/he_seq_%d.png"
                % kk,
                tx[0],
            )
            plt.imsave(
                "/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/0.36/results/200/2502/ihc_seq_%d.png"
                % kk,
                tn[0],
            )
            plt.imsave(
                "/Users/cunyuan/DATA/Kimura/qupath-proj/tiles/0.36/results/200/mask_seq_%d.png"
                % kk,
                tn[0, :, :, 0],
                cmap="gray",
            )
            print(kk)
        avgiou /= kk + 1
        print("avgiou:", avgiou)
        num_all_ = num_positive_ + num_negative_
        print(
            "F.Prec. %3.2f F.Reca. %3.2f \nT.Prec. %3.2f T.Reca. %3.2f\nAcc. %3.2f"
            % (
                num_tn_ / num_npred_,
                num_tn_ / num_negative_,
                num_tp_ / num_pred_,
                num_tp_ / num_positive_,
                (num_tn_ + num_tp_) / (num_negative_ + num_positive_),
            )
        )
        print(
            "Labelling index Pred. %3.2f\nTrue. %3.2f"
            % (num_pred_ / num_all_, num_positive_ / num_all_)
        )
    results = model.predict_generator(testGene, 30, verbose=1)

# %%
