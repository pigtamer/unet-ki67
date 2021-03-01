from configs import *


print(fold[0][0])
print(fold[0][1])

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