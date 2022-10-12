#%%
from utils import *
import tensorflow as tf
from datetime import datetime
import os
import numpy as np
HOME_PATH = "/wd_0/ji"
# train_path = HOME_PATH + "/DATA/TILES_256(1 in 10)"
train_path = HOME_PATH + "/TILES_PAD(256, 256)"
val_path = HOME_PATH + "/TILES_PAD(256, 256)"
# val_path = HOME_PATH + "/TILES_256(1 in 10)"
test_path = "/raid/ji/DATA" + "/KimuraLIpng/"

model_dir = HOME_PATH + "/models57/"

seed = 1

# ------------------ 指定训练·测试图像尺寸 ---------------------
edge_size = 256
target_size = (edge_size, edge_size)
test_size = (2048, 2048)

# ------------------ 指定GPU资源 ---------------------
devices = "0,1,2,3"
# devices = ""
os.environ["CUDA_VISIBLE_DEVICES"] = devices
DEVICES=[
"/gpu:%s"%id for id in devices[::2]# ::2 skips puncs in device string
]

num_gpus=len(DEVICES)

lr = 2.5E-4
lr = lr*num_gpus # 线性scale学习率

# ------------------ 强制设置学习率！！！用后还原！！！ ---------------------
# lr = 3.13E-5

lrstr = "{:.2e}".format(lr)

bs_single = 64
if num_gpus == 0: num_gpus = 1
bs = bs_single*num_gpus
bs_v = bs_single*num_gpus

verbose = 1

checkpoint_period = 5

flag_test = 0
flag_multi_gpu = 0

flag_continue = 0
continue_step = (0, 0)  # start epoch, total epochs trained
initial_epoch = continue_step[0] + continue_step[1]

num_epoches = 101

framework = "hvd-tfk"

# model_name = "deeplabv3"
model_name = "dense121-unet"

loss_name = "bceja"  # focalja, bce, bceja, ja, dice...

id_loocv = 7
# data_name = "kmr-imgnet-loocv%s-noaug"%id_loocv
data_name = "G123-57-PAD-lco"
# data_name = "loocv%s"%id_loocv
# data_name = "lrx16valall_kmr-imgnet-sing%s"%id_loocv
oversampling = 1
# FIXED_STEPS = 1600

test_list=[[6, 11], [12, 17], [0,5], [24, 29], [30, 34], [47, 52], [35, 40], [18, 23], [41, 46]]

cross_fold = [["001", "002", "003", "004", "006", "007", "008", "009"], ["005", "010"]]

fold = {'G1': ['7015','1052','3768','7553','5425','3951','2189','3135','3315','4863','4565','2670','3006','3574','3597','3944','1508','0669','1115'],
'G2': ['5256','6747','8107','1295','2072','2204','3433','7144','1590','2400','6897','1963','2118','4013','4498','0003','2943','3525','2839'],
'G3': ['2502','7930','7885','0790','1904','3235','2730','7883','3316','4640','0003','1883','2913','1559','2280','6018','2124','8132','2850']}
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

# sing = sing_group[id_loocv]
#tr_ids = np.hstack([foldmat.ravel()[:id_loocv], foldmat.ravel()[id_loocv + 1 :]])
# tr_ids = [foldmat.ravel()[sing[1]], foldmat.ravel()[sing[2]]]
tr_ids=foldmat.ravel() # Mixed

#val_ids = [foldmat.ravel()[id_loocv]]
# val_ids = [foldmat.ravel()[sing[0]]]
val_ids = foldmat.ravel()
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
