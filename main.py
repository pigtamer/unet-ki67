from configs import *
from model import *
from data import *
from data_kmr import *
from utils import *
from color_proc import *
from tqdm import tqdm
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
    wsi_ids=tr_ids,
    stains=["HE", "Mask"], #DAB, Mask, HE< IHC
    aug=False,
    target_size=target_size,
    cache=False,
    shuffle_buffer_size=16,
    seed=seed,
    num_shards=1
)
valGene, n_val = load_kmr_tfdata(
    dataset_path=val_path,
    batch_size=bs_v,
    cross_fold=cross_fold[1],
    wsi_ids=val_ids,
    stains=["HE", "Mask"],
    aug=False,
    cache=False,
    shuffle_buffer_size=16,
    seed=seed,
    num_shards=1
)
testGene, n_test = load_kmr_test(
    dataset_path=test_path,
    target_size=(2048, 2048),
    batch_size=1,
    cross_fold=cross_fold[1],
    wsi_ids=[foldmat[0, 2],],
    aug=False,
    cache=False,
    shuffle_buffer_size=128,
    seed=seed,
)
# testGene = testGenerator(test_path, as_gray=False, target_size=target_size)

# ======　检查配对 =======
# for kk, (tx, ty) in zip(range(n_train), trainGene):
#     for img, msk in zip(tx, ty):
#         print(img.numpy().max(), "   ", msk.numpy().max())
#         plt.figure()
#         plt.axis(False)
#         plt.subplot(121);plt.imshow(img)
#         plt.subplot(122);plt.imshow(msk)
#         plt.show()

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
callbacks = [model_checkpoint,tensorboard_callback]

# model = deeplab(loss=loss_name, lr = lr, classes=3)
model = smunet(loss=loss_name)
# model = u_res50enc(loss=loss_name)
# model = unet(loss=loss_name, 
#  	pl=[16,32,64,128,256],
#  	# pl=[256,128,64,32,16], # reverted
#  	)
# model = kumatt(loss=loss_name)

if flag_continue:
    start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_bs%s_ep%02d+%02d" % (
        framework,
        model_name,
        data_name,
        loss_name,
        edge_size,
        lrstr,
        bs,
        continue_step[0],
        continue_step[1]
    )
    for layer in model.layers :
        print(layer.name+" : input ("+str(layer.input_shape)+") output ("+str(layer.output_shape)+")")

    print("__")
    import h5py
    with h5py.File(start_path, 'r') as f:
        for k in f.keys():
            for l in f[k].keys():
                for m in f[k][l].keys():
                    print(k+ " : " + m + " : " + str(f[k][l][m]))
    model.load_weights(start_path, by_name=True)

if not flag_test:
    training_history = model.fit(
        trainGene,
        validation_data=valGene,
        validation_freq=1,
        validation_steps=n_val//bs_v,
        steps_per_epoch=step_num*oversampling,
        epochs=num_epoches,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )
else:
    # ----------------------------------------------------------------------
    # 转换tfpb模型为h5
    # ----------------------------------------------------------------------

    # for k in range(10, 11):
    #     start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
    #         framework,
    #         model_name,
    #         data_name,
    #         loss_name,
    #         edge_size,
    #         lrstr,
    #         continue_step[0] + continue_step[1],
    #         k * checkpoint_period,
    #     )
    #     print(start_path)
    #     model = tf.keras.models.load_model(start_path)
    #     model.save(model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5"% (
    #         framework,
    #         model_name,
    #         data_name,
    #         loss_name,
    #         edge_size,
    #         lrstr,
    #         continue_step[0] + continue_step[1],
    #         k * checkpoint_period,
    #     ))
        #model.evaluate(valGene, steps=n_val//bs_v)
        # kappa = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=True)
        # kappa.update_state(y_true , y_pred)
    for k in range(50, 51):
        # continue each model checkpoint
        start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
            framework,
            model_name,
            data_name,
            loss_name,
            edge_size,
            lrstr,
            # bs,
            continue_step[0] + continue_step[1],
            k,
        )

        # ----------------------------------------------------------------------
        # ROC曲线绘制
        # ----------------------------------------------------------------------
        sm.set_framework('tf.keras')
        model.load_weights(start_path)

        print(n_val//bs_v)
        from sklearn.metrics import roc_curve, auc

        fig = plt.figure(figsize=(10,10), dpi=300)
        plt.plot([0, 1], [0, 1], "k--")
        auclist = []
        legs = ['Luck']
        show_id_list = ["1-1", "1-2", "1-3", 
            "2-1", "2-2", "2-3",
            "3-1", "3-2", "3-3"]
        for id_loocv_t in range(9):
            data_name_dict_t = {"ALL": "ALL",
                        "LOCOCV": "kmr-imgnet-loocv%s-noaug"%id_loocv_t,
                        "SINGLE": "kmr-imgnet-sing%s"%id_loocv_t,}
            data_name_t = data_name_dict_t[scheme]
            start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
                    framework,
                    model_name,
                    data_name_t,
                    loss_name,
                    edge_size,
                    lrstr,
                    continue_step[0] + continue_step[1],
                    k,
                )
            print("AUC test: ", start_path)
            model.load_weights(start_path)
            print(val_ids[id_loocv_t])
            valGene, n_val = load_kmr_tfdata(
                            dataset_path=val_path,
                            batch_size=bs_v,
                            cross_fold=cross_fold[1],
                            wsi_ids=[val_ids[id_loocv_t]],
                            stains=["HE", "Mask"],
                            aug=False,
                            cache=False,
                            shuffle_buffer_size=128,
                            seed=seed,
                        )

            #======== 评估IOU ====================
            model.evaluate(x = valGene, 
                            batch_size = bs_v,
                            verbose=1,
                            steps=n_val//bs_v,
                            callbacks=callbacks,
                            )

            # ======== 评估AUC ====================
        #     for k_val, (x, y) in zip( range(n_val//bs_v), valGene):
        #         f = model.predict(x, batch_size=bs_v, verbose=0)
        #         y = y.numpy().reshape(-1,)[::1000]
        #         f = f.reshape(-1,)[::1000]
        #         if k_val == 0:
        #             Y, F = y, f
        #         else:
        #             Y, F = np.concatenate([Y, y]), np.concatenate([F, f])
        #     print(classification_report(Y > 0, F>0.5))
        #     fpr, tpr, _ = roc_curve(Y.ravel(), F.ravel())
        #     area_under_curve = auc(fpr, tpr)
        #     auclist.append(area_under_curve)
        #     plt.plot(fpr[::1000], tpr[::1000], label="%s"%id_loocv_t+", AUC = {:.3f}".format(area_under_curve))
        #     legs = legs + [show_id_list[id_loocv_t]]
        # plt.xlabel("False positive rate")
        # plt.ylabel("True positive rate")
        # plt.title("ROC curve")


        # for k in range(1, len(legs)): legs[k] += ", {:.3f}".format(auclist[k-1])
        
        # plt.legend(legs, loc="best")
        # plt.tight_layout()
        # plt.grid()

        # plt.savefig("/home/cunyuan/roc.png")

        num_tp_, num_tn_, num_pred_, num_npred_, num_positive_, num_negative_ = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        avgiou = 0
        for id_loocv_t in range(8,9):
            # data_name_t = "kmr-imgnet-loocv%s-noaug"%id_loocv_t
            data_name_t = "kmr-imgnet-%s%s"%(scheme, id_loocv_t)
            start_path = model_dir + "%s-%s__%s_%s_%d_lr%s_ep%02d+%02d.h5" % (
                    framework,
                    model_name,
                    data_name_t,
                    loss_name,
                    edge_size,
                    lrstr,
                    # bs,
                    continue_step[0] + continue_step[1],
                    k,
                )
            model.load_weights(start_path)
            print(start_path)
            testGene, n_test = load_kmr_test(
                                dataset_path=test_path,
                                target_size=(2048, 2048),
                                batch_size=1,
                                cross_fold=cross_fold[1],
                                wsi_ids=[foldmat[0, 2],],
                                aug=False,
                                cache=False,
                                shuffle_buffer_size=128,
                                seed=seed,
                            )
            for kk, (tx, ty) in zip(range(n_test), testGene):
                if data_name != "ALL":
                    if kk< test_list[id_loocv_t][0]: continue
                    if kk > test_list[id_loocv_t][1]: break
            
            # tn = ty
            # (
            #     num_tp,
            #     num_tn,
            #     num_pred,
            #     num_npred,
            #     num_positive,
            #     num_negative,
            #     iou,
            #     res,
            # ) = single_prediction(tx, ty, tn, model, None, 256)
            # num_tp_ += num_tp
            # num_tn_ += num_tn
            # num_pred_ += num_pred
            # num_npred_ += num_npred
            # num_positive_ += num_positive
            # num_negative_ += num_negative
            # avgiou += iou
            # plt.show()
                res, hema_texture, mask = interactive_prediction(tx[0, :,:,:3], model)
                plt.imsave(
                    "/raid/ji/DATA/KimuraLIpng/ihc_%d.png"
                    % kk,
                    res.reshape(2048, 2048, 3),
                )
            # plt.imsave(
            #     "/home/cunyuan/resdenseunet/he_%d.png"
            #     % kk,
            #     tx.numpy()[:,:,:,:3].reshape(2048, 2048, 3),
            # )
