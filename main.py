from configs import *
from model import *
from data import *
from data_kmr import *
from utils import *
from color_proc import *
from tqdm import tqdm
import segmentation_models as sm

sm.set_framework("tf.keras")


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
)


if not flag_test:
    trainGene_n, valGene_n = load_kmr57_tfdata(
        dataset_path=train_path,
        batch_size=bs,
        cross_fold=["*"],
        wsi_ids=tr_ids,
        # wsi_ids=['7015','1052','3768','5256','6747','8107','2502','7930','7885'],
        # wsi_ids=['*'],
        stains=["HE", "Mask"],  # DAB, Mask, HE< IHC
        aug=False,
        target_size=target_size,
        cache=False,
        shuffle_buffer_size=5000,
        seed=seed,
        num_shards=1,
    )
    if not ("ALL" in data_name):
        valGene_n, _ = load_kmr57_tfdata(
            dataset_path=train_path,
            batch_size=bs,
            cross_fold=["*"],
            wsi_ids=val_ids,
            stains=["HE", "Mask"],  # DAB, Mask, HE< IHC
            aug=False,
            target_size=target_size,
            cache=False,
            shuffle_buffer_size=5000,  
            seed=seed,
            num_shards=1,
        )
    trainGene, n_train = trainGene_n
    valGene, n_val = valGene_n

    print("NUM TRAIN:", n_train)
    step_num = n_train // bs
    # step_num=10 # for test

    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor="loss",
        verbose=verbose,
        save_best_only=False,
        save_weights_only=True,
        mode="auto",
        save_freq=checkpoint_period * step_num * oversampling,
    )

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = TensorBoard(log_dir=logdir)
    callbacks = [model_checkpoint, tensorboard_callback]

    # model = deeplab(loss=loss_name, lr = lr, classes=3)
    model = smunet(loss=loss_name)
    # model = u_res50enc(loss=loss_name)
    # model = unet(loss=loss_name,
    #  	pl=[16,32,64,128,256],
    #  	# pl=[256,128,64,32,16], # reverted
    #  	)
    # model = kumatt(loss=loss_name)
    training_history = model.fit(
        trainGene,
        validation_data=valGene,
        validation_freq=1,
        validation_steps=n_val // bs_v,
        steps_per_epoch=step_num * oversampling,
        epochs=num_epoches,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )
else:
    model = smunet(loss=loss_name)

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
    # model.evaluate(valGene, steps=n_val//bs_v)
    # kappa = tfa.metrics.CohenKappa(num_classes=2, sparse_labels=True)
    # kappa.update_state(y_true , y_pred)
    for k in range(10, 10 + 1):
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
        # sm.set_framework('tf.keras')
        # model.load_weights(start_path)

        # print(n_val//bs_v)
        # from sklearn.metrics import roc_curve, auc

        # fig = plt.figure(figsize=(10,10), dpi=300)
        # plt.plot([0, 1], [0, 1], "k--")
        # auclist = []
        # for cases in ["All cases"]:
        #     for cases in [foldmat[2, 2],]:
        #         valGene, n_val = load_kmr_tfdata(
        #                         dataset_path=val_path,
        #                         batch_size=bs_v,
        #                         cross_fold=cross_fold[1],
        #                         # wsi_ids=foldmat.ravel(),
        #                         # wsi_ids=np.hstack([foldmat[1, :]]).ravel(),
        #                         wsi_ids=[cases],
        #                         stains=["HE", "Mask"],
        #                         aug=False,
        #                         cache=False,
        #                         shuffle_buffer_size=128,
        #                         seed=seed,
        #                     )
        #         for k_val, (x, y) in zip( tqdm(range(n_val//bs_v)), valGene):
        #         # for k_val, (x, y) in zip( tqdm(range(1)), valGene):
        #             f = model.predict(x, batch_size=bs_v)
        #             # plt.show()
        #             # print(k_val)
        #             y = y.numpy().reshape(-1,)[::100]
        #             f = f.reshape(-1,)[::100]
        #             # print(classification_report(y > 0, f>0.5))
        #             if k_val == 0:
        #                 Y, F = y, f
        #                 # thresh_argmax_f1 = 0
        #                 # print(start_path)
        #                 # print("Model @ epoch %d" % (k * checkpoint_period), "\n", "-*-" * 10)
        #             else:
        #                 Y, F = np.concatenate([Y, y]), np.concatenate([F, f])
        #         print(classification_report(Y > 0, F>0.5))
        #         np.savetxt("%s.csv"%cases, [Y, F], delimiter=",")
        #         fpr, tpr, _ = roc_curve(Y.ravel(), F.ravel())
        #         area_under_curve = auc(fpr, tpr)
        #         auclist.append(area_under_curve)
        #         plt.plot(fpr, tpr, label="AUC = {:.3f}".format(area_under_curve))
        #         plt.xlabel("False positive rate")
        #         plt.ylabel("True positive rate")
        #         plt.title("ROC curve")
        # legs = ['Luck']+['All cases']
        # # legs = ['Luck']+[x[6:10] for x in foldmat.ravel()]
        # for k in range(1, len(legs)): legs[k] += ", {:.3f}".format(auclist[k-1])
        # plt.legend(legs, loc="best")
        # plt.tight_layout()
        # plt.grid()
        # # plt.show()
        # plt.savefig("/home/cunyuan/roc.png")
        # exit(0)

        # ----------------------------------------------------------------------
        # 残留代码，决定最佳阈值
        # ----------------------------------------------------------------------
        """
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
        """
        num_tp_, num_tn_, num_pred_, num_npred_, num_positive_, num_negative_ = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
        avgiou = 0
        for id_loocv_t in range(18):
            # data_name_t = "kmr-imgnet-loocv%s-noaug"%id_loocv_t
            data_name_t = data_name
            # data_name_t = "kmr-imgnet-loocv%s"%id_loocv_t
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
            start_path = (
                "/wd_0/ji/_MODELS/fcv/"
                + "hvd-tfk-dense121-unet__G123-57-cv%s_bceja_256_lr1.00e-03_ep00+10.h5"
                % id_loocv_t
            )
            start_path = (
                model_dir
                + "hvd-tfk-dense121-unet__G123-57-sing%s_bceja_256_lr1.00e-03_ep00+10.h5"
                % id_loocv_t
            )
            # cvid = int(id_loocv_t*3)
            cvid = int(id_loocv_t)
            # model.build()
            # import h5py
            # with h5py.File(start_path, 'r') as f:
            #     for k in f.keys():
            #         for l in f[k].keys():
            #             for m in f[k][l].keys():
            #                 print(k+ " : " + m + " : " + str(f[k][l][m]))
            # model.load_weights(start_path, by_name=True)
            # model.build(input_shape=(None, 256, 256, 3))
            model.load_weights(start_path)
            print(start_path)
            
            sing = sing_group[cvid % 6]

            if "sing" in data_name:
                tr_ids = list(foldmat[cvid // 6][:3*sing[0]]) + list(foldmat[cvid // 6][3*(sing[0]+1):])
                val_ids = foldmat[cvid // 6][3*sing[0]:3*(sing[0]+1)]
                if cvid%6 == 5:
                    tr_ids = list(foldmat[cvid // 6][:3*sing[0]])
                    val_ids = foldmat[cvid // 6][3*sing[0]:]
            elif "cv" in data_name:
                tr_ids= fold["G1"][:cvid] + fold["G2"][:cvid] + fold["G3"][:cvid] + fold["G1"][cvid+3:] + fold["G2"][cvid+3:] + fold["G3"][cvid+3:]
                val_ids = fold["G1"][cvid:cvid+3] + fold["G2"][cvid:cvid+3] + fold["G3"][cvid:cvid+3]
                if cvid//3 == 5:
                    tr_ids = fold["G1"][:cvid] + fold["G2"][:cvid] + fold["G3"][:cvid]
                    val_ids = fold["G1"][cvid:] + fold["G2"][cvid:] + fold["G3"][cvid:]
            elif "ALL" in data_name:
                tr_ids = ["*"]
                val_ids = ["*"]
            testGene, n_test = glob_kmr_test(
                dataset_path=test_path,
                target_size=(2048, 2048),
                batch_size=1,
                cross_fold=None,
                # wsi_ids=fold["G1"][cvid:cvid+3] + fold["G2"][cvid:cvid+3] + fold["G3"][cvid:cvid+3],
                wsi_ids = val_ids,
                aug=False,
                cache=False,
                shuffle_buffer_size=128,
                seed=seed,
            )
            for kk, (tx, ty) in zip(range(n_test), testGene):
                # if data_name != "ALL":
                ## For old eval code
                #     if kk< test_list[id_loocv_t][0]: continue
                #     if kk > test_list[id_loocv_t][1]: break
                txl = tx
                print(txl)
                tx = cv.imread(tx)
                tx = cv.cvtColor(tx, cv.COLOR_BGR2RGB)
                tx = np.expand_dims(tx, 0) / 255.0
                res, hema_texture, mask = interactive_prediction(tx[0, :, :, :3], model)
                plt.imsave(
                    "/wd_0/ji/DATA/2210-WSI57-SING/%s.png" % txl.rsplit("/", 1)[-1],
                    res,
                )
            # plt.imsave(
            #     "/home/cunyuan/resdenseunet/he_%d.png"
            #     % kk,
            #     tx.numpy()[:,:,:,:3].reshape(2048, 2048, 3),
            # )
