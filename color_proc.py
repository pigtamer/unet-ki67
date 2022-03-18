import matplotlib.pyplot as plt
import cv2 as cv
from numpy import *
# from skimage.color import rgb2hed, hed2rgb, separate_stains, combine_stains
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D  # --- For 3D surface drawing
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score


# https://github.com/scikit-image/scikit-image/blob/0e8cff47a533e240508e2485032dac78220ac6c9/skimage/color/colorconv.py#L1375

def norm_by_row(M):
    for k in range(M.shape[1]):
        M[k, :] /= np.sqrt(np.sum(np.power(M[k, :], 2)))
    return M


def showbychan(im_ihc):
    for k in range(3):
        plt.figure()
        plt.imshow(im_ihc[:, :, k], cmap="gray")


def rgbdeconv(rgb, conv_matrix, C=0):
    rgb = rgb.copy().astype(float)
    rgb += C
    stains = np.reshape(-np.log10(rgb), (-1, 3)) @ conv_matrix
    return np.reshape(stains, rgb.shape)


def hecconv(stains, conv_matrix, C=0):
    stains = stains.astype(float)
    logrgb2 = -np.reshape(stains, (-1, 3)) @ conv_matrix
    rgb2 = np.power(10, logrgb2)
    return np.reshape(rgb2 - C, stains.shape)


def surf(matIn, div=(1, 1), SIZE=(8, 6)):
    x = np.arange(0, matIn.shape[0])
    y = np.arange(0, matIn.shape[1])
    x, y = np.meshgrid(y, x)
    fig = plt.figure(figsize=SIZE)
    ax = Axes3D(fig)
    ax.plot_surface(x, y, matIn, rstride=div[0], cstride=div[1], cmap='hot')
    plt.title('fig')
    plt.show()


def cntAna(mask_father, mask_son):
    """

    Parameters
    ----------
    mask_father
    mask_son

    Returns
    -------

    """
    c0, h0 = cv.findContours(mask_father.astype(uint8), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_TC89_KCOS)
    c1, h1 = cv.findContours(mask_son.astype(uint8), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_TC89_KCOS)
    kk = 0
    if len(c0) * len(c1) == 0:
        return 0
    else:
        for j in range(len(c0)):
            for k in range(len(c1)):
                # check if c0 has sons in c1
                if (cv.pointPolygonTest(c0[j], tuple(c1[k][0, 0]), False) >= 0):
                    flag = 1
                    break
                else:
                    flag = 0
            kk += flag
        return kk


# The definition of trans. matrix in stain separation article

H_DAB = array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78]
])

H_Mou = H_DAB.copy()
H_Mou[2, :] = np.cross(H_DAB[0, :], H_DAB[1, :])
H_Mou = norm_by_row(H_Mou)
H_Mou_inv = norm_by_row(np.linalg.inv(H_Mou))

H_ki67 = H_DAB.copy()
H_ki67[1, :] = np.cross(H_DAB[0, :], H_DAB[2, :])
H_ki67 = norm_by_row(H_ki67)
H_ki67_inv = norm_by_row(np.linalg.inv(H_ki67))
# color maps for visualization

cmap_hema = LinearSegmentedColormap.from_list('cmap_hema', ['white', 'navy'])
cmap_eosin = LinearSegmentedColormap.from_list('cmap_eosin', ['white', 'darkviolet'])
cmap_dab = LinearSegmentedColormap.from_list('cmap_dab', ['white', 'saddlebrown'])


def single_prediction(im_in, label, nuclei, net, li_mask=None, net_sizein=256):
    W, H = im_in.shape[1], im_in.shape[2]
    if W % net_sizein != 0 or H % net_sizein != 0:
        raise ValueError("")
    # TODO: 增加对padding的支持，图幅不一定是输入的整数倍
    w_num, h_num = W // net_sizein, H // net_sizein
    res = zeros((W, H, 3))
    res_set = zeros((W, H, 3))
    avgiou = 0;
    iou = 0;
    f1 = 0
    pprecision = 0
    precall = 0
    acc_regional = 0
    kp, ka = 0, 0
    num_all, num_pred, num_positive, num_tp = 0, 0, 0, 0
    for i in range(w_num):
        for j in range(h_num):
            if li_mask is not None:
                li_mask_chip = li_mask[i * net_sizein:(i + 1) * net_sizein,
                    j * net_sizein:(j + 1) * net_sizein].reshape((1, net_sizein, net_sizein, 1)) / 255
            else:
                li_mask_chip = np.ones((1, 256,256,1))

            chip = im_in[:, i * net_sizein:(i + 1) * net_sizein,
                   j * net_sizein:(j + 1) * net_sizein,
                   :3].numpy().reshape(1, net_sizein, net_sizein, 3)*li_mask_chip
            dchip = label[0, i * net_sizein:(i + 1) * net_sizein,
                    j * net_sizein:(j + 1) * net_sizein,
                    0].numpy().reshape(1, net_sizein, net_sizein, 1)*li_mask_chip
            nchip = nuclei[0, i * net_sizein:(i + 1) * net_sizein,
                    j * net_sizein:(j + 1) * net_sizein,
                    0].numpy().reshape(1, net_sizein, net_sizein, 1) / 255*li_mask_chip

            mask = net.predict(chip)[0, :, :, 0].reshape(1, net_sizein, net_sizein, 1)*li_mask_chip

            # iou += jaccard_score(dchip.reshape(-1, ) > 0, mask.reshape(-1, ) > 0.6)
            # f1 += f1_score(dchip.reshape(-1, ) > 0, mask.reshape(-1, ) > 0.6)
            chip = chip[0, :, :, :]
            nchip = nchip[0, :, :, 0]
            dchip = dchip[0, :, :, 0]
            mask = mask[0,:,:,0]
            """
            Morphology regional evaluation
            """
            nuclei_=0
            label_ = 0
            pred_ = 0
            # exkernel = np.ones((5, 5), np.uint8)
            # nuclei_ = cv.morphologyEx(nchip, cv.MORPH_OPEN, exkernel).astype(np.uint8)
            # label_ = dchip.astype(np.uint8)
            # pred_ = (mask > 0.6).astype(np.uint8)
            # pred_ = cv.morphologyEx(pred_, cv.MORPH_OPEN, exkernel)
            # label_ = label_ * nuclei_

            # tp_map = (label_ * pred_).astype(np.uint8)
            # pred_map = (nuclei_ * pred_).astype(np.uint8)

            # num_tp += cntAna(label_, tp_map)
            # num_positive += cntAna(nuclei_, label_)
            # num_pred += cntAna(nuclei_, pred_map)

            # ncts, _ = cv.findContours(tp_map, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            # ncts_tp, _ = cv.findContours(tp_map, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            """
            ncts_allnuc, _ = cv.findContours(nuclei_, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            """
            ncts_allnuc = []
            # ncts_lbl, _ = cv.findContours(label_, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            # ncts_pred, _ = cv.findContours(pred_, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

            num_all += len(ncts_allnuc)
            # num_positive += len(ncts_lbl)
            # num_pred += len(ncts_pred)
            # num_tp += len(ncts_tp)
            num_npred = num_all - num_pred
            num_negative = num_all - num_positive
            num_tn = num_all - (num_positive + num_pred - num_tp)
            num_fn = num_positive - num_tp
            num_fp = num_pred - num_tp

            # if num_all != 0:
            #     ka += 1
            #     if num_positive != 0 and num_pred != 0:
            #         kp += 1
                    # pprecision += num_tp / num_pred
                    # precall += num_tp / num_positive
                # elif num_positive == 0:
                #     print("No positive nuclei in view")
                # elif num_pred == 0:
                #     print("Did not detect ki67+ in labels")

                # acc_regional += (num_tp + num_tn) / num_all
                # print(kp, ka)
                # print("Overall acc%f\n" % (acc_regional / ka))

            hema_texture = rgbdeconv(chip, H_Mou_inv, C=0)[:, :, 0]
            pseudo_dab = hema_texture * mask

            res[i * net_sizein:(i + 1) * net_sizein,
            j * net_sizein:(j + 1) * net_sizein,
            -1] = pseudo_dab
            res[i * net_sizein:(i + 1) * net_sizein,
            j * net_sizein:(j + 1) * net_sizein,
            0] = hema_texture

            # res_set[i * net_sizein:(i + 1) * net_sizein,
            # j * net_sizein:(j + 1) * net_sizein] = nuclei_ * 2 + label_ * 4 + pred_ * 8 + tp_map * 16

            # colored policy
            res_set[i * net_sizein:(i + 1) * net_sizein,
            j * net_sizein:(j + 1) * net_sizein, 0] = label_ * 255
            res_set[i * net_sizein:(i + 1) * net_sizein,
            j * net_sizein:(j + 1) * net_sizein, 1] = pred_ * 255
            res_set[i * net_sizein:(i + 1) * net_sizein,
            j * net_sizein:(j + 1) * net_sizein, 2] = nuclei_ * 255

    # if num_pred != 0 and num_positive != 0 and num_all != 0:
    #     pprecision = num_tp / num_pred
    #     precall = num_tp / num_positive
    #     acc_regional = (num_tp + num_tn) / num_all

        # print("=-=" * 10)
        # print("Overall acc%f" % (acc_regional))
        # print("Precision: %f\nRecall %f\n" % (
        #     pprecision, precall))
    # lbi = num_pred / (num_all + 1E-6)
    # lbi_true = num_positive / (num_all+1E-6)
    # print("---" * 10, "\nLabelling index: [True] %3.2f [Ours] %3.2f" % (lbi_true, lbi))
    # plt.figure(figsize=(6, 6))
    # plt.imshow(res_set);
    # plt.axis('off');
    # plt.title("Region # policy Prec. %3.2f Rec. %3.2f\nLabelling index: [True] %3.2f [Ours] %3.2f" %
    #           (pprecision, precall, lbi_true, lbi))

    iou /= (w_num * h_num)
    f1 /= w_num * h_num
    # print(iou, f1)
    res = hecconv(res, H_ki67)
    res = np.clip(res, 0, 1)
    return (num_tp, num_tn, num_pred, num_npred, num_positive, num_negative, iou, res)

def interactive_prediction(im_in, net, net_sizein=256):
    W, H = im_in.shape[0], im_in.shape[1]
    print("WH", W, H)
    S = net_sizein
    im_padded = np.ones((W+S*2, H+S*2, 3))
    im_padded[S:S+W, S:S+H, :] = im_in
    im_in = im_padded
    res = zeros((W, H, 3))

    W += 2*S
    H += 2*S

    mask = zeros((W, H))
    maskop = zeros((W, H))

    r = 8
    pw, ph = S//r, S//r

    w_num, h_num = (W -2*pw) // ((r-1)*pw), (H -2*ph) // ((r-1)*ph)
    bchip =  np.zeros((w_num*h_num, S, S, 3))

    # make batches for efficient GPU computation
    for i in range(w_num):
        for j in range(h_num):
            bchip[i*h_num + j, :, :, :] = im_in[i * (r-1)*pw:(i) * (r-1)*pw + S,
                   j * (r-1)*ph:(j)*(r-1)*ph + S,
                   :].reshape(1, S, S, 3)

    bmask = net.predict(bchip)
    print(bmask.shape)
    for i in range(w_num):
        for j in range(h_num):
            mask[i * (r-1)*pw:(i) * (r-1)*pw + S, j * (r-1)*ph:(j)*(r-1)*ph + S] += \
                bmask[i*h_num + j, :, :, 0]
            maskop[i * (r-1)*pw:(i) * (r-1)*pw + S, j * (r-1)*ph:(j)*(r-1)*ph + S] += np.ones_like(bmask[i*h_num + j, :, :, 0])     
    mask = mask / maskop
    mask[mask>0.5]=1
    mask[mask<=0.5]=0
    mask = mask[S:W-S, S:H-S]
    hema_texture = rgbdeconv(im_in[S:W-S, S:H-S, :], H_Mou_inv, C=0)[:, :, 0]
    print(hema_texture.shape, mask.shape, res.shape)
    pseudo_dab = (hema_texture * mask)
    res[:,:,-1] = pseudo_dab
    res[:,:,0] = hema_texture
    res = hecconv(res, H_ki67)
    res = np.clip(res, 0, 1)
    res = res # crop valid region
    return res, hema_texture, mask
