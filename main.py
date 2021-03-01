from configs import *
from model import *
from data import *
from data_kmr import *
from utils import *
from color_proc import *

import segmentation_models as sm; sm.set_framework("tf.keras")


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
)

trainGene, n_train = load_kmr_tfdata(
    dataset_path=train_path,
    batch_size=bs,
    cross_fold=cross_fold[0],
    wsi_ids=fold[0][0],
    aug=False,
    target_size=target_size,
    cache=False,
    shuffle_buffer_size=128,
    seed=seed,
)
valGene, n_val = load_kmr_tfdata(
    dataset_path=val_path,
    batch_size=bs_v,
    cross_fold=cross_fold[1],
    wsi_ids=fold[0][0],
    aug=False,
    cache=False,
    shuffle_buffer_size=128,
    seed=seed,
)

testGene = testGenerator(test_path, as_gray=False, target_size=target_size)

model = smunet(loss=loss_name)

step_num = n_train // bs
training_history = model.fit(
    trainGene,
    validation_data=valGene,
    validation_freq=5,
    validation_steps=100,
    steps_per_epoch=step_num,
    epochs=num_epoches,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
)
