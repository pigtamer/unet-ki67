#%%
import horovod.tensorflow.keras as hvd
hvd.init()
from utils import *
import tensorflow as tf
from datetime import datetime

HOME_PATH = "/gs/hs0/tga-yamaguchi.m/ji"
train_path = HOME_PATH + "/DATA/TILES_(256, 256)"
val_path = HOME_PATH + "/DATA/TILES_(256, 256)"
test_path = HOME_PATH + "/DATA/test_1024/k"

model_dir = HOME_PATH + "/models/"

seed = 1

edge_size = 256
target_size = (edge_size, edge_size)
test_size = (2048, 2048)

# ------------------ 强制设置学习率！！！用后还原！！！ ---------------------
lr = 1E-3 / hvd.size()

lrstr = "{:.2e}".format(lr)

bs = 16
bs_v = 16
verbose = 1

checkpoint_period = 5

flag_test = 0
flag_multi_gpu = 0

flag_continue = 0
continue_step = (0, 0)  # start epoch, total epochs trained
initial_epoch = continue_step[0] + continue_step[1]

num_epoches = 55

framework = "hvd-tfk"

# model_name = "deeplabv3"
model_name = "dense121-unet"

loss_name = "bceja"  # focalja, bce, bceja, ja, dice...

id_loocv = 3
data_name = "kmr-imgnet-loocv%s-noaug"%id_loocv
oversampling = 1

configstring = "%s_%s_%s_%s_%d_lr%s_bs%sxn%s" % (
    framework,
    model_name,
    data_name,
    loss_name,
    edge_size,
    lrstr,
    bs,
    hvd.size()
)
print(configstring)

fold = folds(
    l_wsis=[
        k + ""
        for k in [
            "01_14-7015_Ki67",  # 1 22091
            "01_17-5256_Ki67",  # 2 54923
            "01_17-7885_Ki67",  # 3 42635 --> 466272
            "01_15-1052_Ki67",  # 1 34251
            "01_17-6747_Ki67",  # 2 66635
            "01_15-2502_Ki67",  # 3 136715
            "01_14-3768_Ki67",  # 1 106379
            "01_17-8107_Ki67",  # 2 69097
            "01_17-7930_Ki67",  # 3 53195
        ]
    ],
    k=3,
)
cross_fold = [["001", "002", "003", "004",  "006", "007", "008", "009"], ["005", "010"]]

fold = {
    "G1": ["01_14-7015_Ki67", "01_15-1052_Ki67", "01_14-3768_Ki67"],
    "G2": ["01_17-5256_Ki67", "01_17-6747_Ki67", "01_17-8107_Ki67"],
    "G3": ["01_17-7885_Ki67", "01_15-2502_Ki67", "01_17-7930_Ki67"],
}
foldmat = np.vstack([fold[key] for key in fold.keys()])

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

logdir = (
    HOME_PATH+ "/logs/scalars/"
    + datetime.now().strftime("%Y%m%d-%H%M%S")
    + configstring
)

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
# %%
