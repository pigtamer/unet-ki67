#%%
from utils import *
import tensorflow as tf
from datetime import datetime
import os

scheme = "ALL"

HOME_PATH = "/raid/ji"
# train_path = HOME_PATH + "/DATA/TILES_256(1 in 10)"
train_path = HOME_PATH + "/DATA/TILES_(256, 256)"
val_path = HOME_PATH + "/DATA/TILES_(256, 256)"
# val_path = HOME_PATH + "/DATA/TILES_256(1 in 10)"
test_path = HOME_PATH + "/DATA/KimuraLIpng/"

model_dir = HOME_PATH + "/ep50models/"+scheme+"/ep50/"

seed = 1
flag_test = 0

# ------------------ 指定训练·测试图像尺寸 ---------------------
edge_size = 256
target_size = (edge_size, edge_size)
test_size = (2048, 2048)

# ------------------ 指定GPU资源 ---------------------
devices = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = devices
DEVICES=[
"/gpu:%s"%id for id in devices[::2]# ::2 skips puncs in device string
]

num_gpus=len(DEVICES)

lr = 1E-3
lr = lr*num_gpus # 线性scale学习率

# ------------------ 强制设置学习率！！！用后还原！！！ ---------------------
# lr = 3.13E-5

lrstr = "{:.2e}".format(lr)

bs_single = 64
bs = bs_single*num_gpus
bs_v = bs_single*num_gpus
verbose = 1

checkpoint_period = 5

flag_test = 0
flag_multi_gpu = 0

flag_continue = 0
continue_step = (0, 0)  # start epoch, total epochs trained
initial_epoch = continue_step[0] + continue_step[1]

num_epoches = 51

framework = "hvd-tfk"

# model_name = "deeplabv3"
model_name = "dense121-unet"
# model_name = "unet-reduce"

loss_name = "l1"  # focalja, bce, bceja, ja, dice...

id_loocv = 7
data_name_dict = {"ALL": "ALL",
            "LOCOCV": "kmr-imgnet-loocv%s-noaug"%id_loocv,
            "SINGLE": "kmr-imgnet-sing%s"%id_loocv,}
data_name = "ihc" + data_name_dict[scheme]

oversampling = 1
# FIXED_STEPS = 1600

test_list=[[6, 11], [12, 17], [0,5], [24, 29], [30, 34], [47, 52], [35, 40], [18, 23], [41, 46]]

cross_fold = [["001", "002", "003", "004", "006", "007", "008", "009"], ["005", "010"]]

fold = {
    "G1": ["01_14-7015_Ki67", "01_15-1052_Ki67", "01_14-3768_Ki67"],
    "G2": ["01_17-5256_Ki67", "01_17-6747_Ki67", "01_17-8107_Ki67"],
    "G3": ["01_17-7885_Ki67", "01_15-2502_Ki67", "01_17-7930_Ki67"],
}
foldmat = np.vstack([fold[key] for key in fold.keys()])

sing_group = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
    [3, 4, 5],
    [4, 5, 3],
    [5, 3, 4],
    [6, 7, 8],
    [7, 8, 6],
    [8, 6, 7],
]

if scheme == "LOOCV":
    tr_ids = np.hstack([foldmat.ravel()[:id_loocv], foldmat.ravel()[id_loocv + 1 :]])
    val_ids = [foldmat.ravel()[id_loocv]]
elif scheme == "SING":
    sing = sing_group[id_loocv]
    tr_ids = [foldmat.ravel()[sing[1]], foldmat.ravel()[sing[2]]]
    val_ids = [foldmat.ravel()[sing[0]]]
elif scheme == "ALL":
    tr_ids=foldmat.ravel() # Mixed
    val_ids = foldmat.ravel()
else:
    print("Invalid training scheme")
    exit(1)


print(tr_ids)
print(val_ids)

configstring = "%s_%s_%s_%s_%d_lr%s_bs%sxn%s" % (
    framework,
    model_name,
    data_name,
    loss_name,
    edge_size,
    lrstr,
    bs,
    num_gpus,
)
print(configstring)


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
    HOME_PATH
    + "/logs/scalars/"
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
