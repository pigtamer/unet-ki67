from keras import backend as K
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# focal loss

def focal_loss(gamma=2., alpha=.25):
    import tensorflow as tf

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is useful for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def jaccard_metric(y_true, y_pred):
    eps = 1E-10
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (sum_ - intersection + eps)
    return jac + (sum == intersection)


def ac_loss(lambdaP=1, w=1, size=(128, 128)):
    # active contour loss implemented by chen et.al.
    def Active_Contour_Loss(y_true, y_pred):
        # y_pred = K.cast(y_pred, dtype = 'float64')

        x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal and vertical directions
        y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

        delta_x = x[:, :, 1:, :-2] ** 2
        delta_y = y[:, :, :-2, 1:] ** 2
        delta_u = K.abs(delta_x + delta_y)

        epsilon = 1E-10  # where is a parameter to avoid square root is zero in practice.
        length = w * K.sum(K.sqrt(delta_u + epsilon))  # equ.(11) in the paper

        """
        region term
        """

        C_1 = np.ones(size)
        C_2 = np.zeros(size)

        region_in = K.abs(K.sum(y_pred[:, 0, :, :] * ((y_true[:, 0, :, :] - C_1) ** 2)))  # equ.(12) in the paper
        region_out = K.abs(K.sum((1 - y_pred[:, 0, :, :]) * ((y_true[:, 0, :, :] - C_2) ** 2)))  # equ.(12) in the paper

        loss = length + lambdaP * (region_in + region_out)

        return loss / (size[0] * size[1])

    return Active_Contour_Loss


def roc(by, bf, thresh=0):
    plt.figure(figsize=(10, 10))
    bc = 0
    acc, maxacc = 0, 0
    for y, f in zip(by, bf):
        y = y.reshape(-1, ).round()
        f = f.reshape(-1, )  # silly. delete later

        f *= f > thresh
        ty = np.argsort(f)[::-1]
        mp, mn = sum(y), y.size - sum(y)
        if mp / y.size < 1E-3: continue  # 当GT全部是反例时，忽略这一张
        bc += 1
        xn, yn = 0, 0
        xnl, ynl = np.zeros(ty.shape), np.zeros(ty.shape)
        tmp = 0
        for k in ty:
            xnl[k], ynl[k] = xn, yn
            if y[k] == 0:
                xn += 1 / mn
            else:
                yn += 1 / mp
            tmp += 0.5 * (xn - xnl[k]) * (yn + ynl[k])
        acc += tmp
        if tmp > maxacc: maxacc = tmp
        plt.scatter(xnl, ynl, alpha=0.5, s=2)
    acc /= bc
    plt.title("avg=%3.3f, max=%3.3f" % (acc, maxacc))
    # plt.savefig("roc.png") # saving as svg is very time-consuming for the large scale of point cloud
    return acc
