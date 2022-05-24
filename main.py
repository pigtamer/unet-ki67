from configs import *
from model import *
from data import *
from data_kmr import *
from utils import *
from color_proc import *

import segmentation_models as sm

sm.set_framework("tf.keras")


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
)

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# print(hvd.rank(), ":", foldmat.ravel()[:-1])
# print(foldmat.ravel())

trainGene, n_train = load_kmr_tfdata(
    dataset_path=train_path,
    batch_size=bs,
    cross_fold=cross_fold[0],
    wsi_ids=tr_ids,
    stains=["HE", "Mask"],  # DAB, Mask, HE< IHC
    aug=False,
    target_size=target_size,
    cache=False,
    shuffle_buffer_size=6000,
    seed=seed,
    # num_shards=1
)
valGene, n_val = load_kmr_tfdata(
    dataset_path=val_path,
    batch_size=bs_v,
    cross_fold=cross_fold[1],
    wsi_ids=val_ids,
    stains=["HE", "Mask"],
    aug=False,
    cache=False,
    shuffle_buffer_size=1000,
    seed=seed,
    # num_shards=1
)

testGene = testGenerator(test_path, as_gray=False, target_size=target_size)

model = smunet(loss=loss_name)
print(n_train)
step_num = n_train // bs
# step_num=10 # for test

model_checkpoint = ModelCheckpoint(
    model_path,
    monitor="loss",
    verbose=verbose,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq=checkpoint_period * step_num,
)

file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(log_dir=logdir)
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(
        warmup_epochs=5, initial_lr=lr, verbose=verbose
    ),
]

if hvd.rank() == 0:
    callbacks.append(model_checkpoint)
    callbacks.append(tensorboard_callback)

training_history = model.fit(
    trainGene,
    validation_data=valGene,
    validation_freq=1,
    validation_steps=n_val // bs_v,
    # steps_per_epoch=step_num*oversampling,
    steps_per_epoch=FIXED_STEPS,
    epochs=num_epoches,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
)
