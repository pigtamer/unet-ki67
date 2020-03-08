import matplotlib.pyplot as plt
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


def single_prediction(im_in, label, nuclei, net, net_sizein):
    W, H = im_in.shape[1], im_in.shape[2]
    if W % net_sizein != 0 or H % net_sizein != 0:
        raise ValueError("")
    # TODO: 增加对padding的支持，图幅不一定是输入的整数倍
    w_num, h_num = W // net_sizein, H // net_sizein
    res = zeros((W, H, 3))
    avgiou = 0;
    iou = 0;
    f1 = 0
    for i in range(w_num):
        for j in range(h_num):
            chip = im_in[:, i * net_sizein:(i + 1) * net_sizein,
                   j * net_sizein:(j + 1) * net_sizein,
                   :]
            dchip = label[0, i * net_sizein:(i + 1) * net_sizein,
                    j * net_sizein:(j + 1) * net_sizein,
                    0]
            nchip = nuclei[0, i * net_sizein:(i + 1) * net_sizein,
                    j * net_sizein:(j + 1) * net_sizein,
                    0] / 255
            mask = net.predict(chip)[0, :, :, 1]
            iou += jaccard_score(dchip.reshape(-1, ) > 0, mask.reshape(-1, ) > 0.6)
            f1 += f1_score(dchip.reshape(-1, ) > 0, mask.reshape(-1, ) > 0.6)
            chip = chip[0, :, :, :]
            # hema_texture = rgbdeconv(chip, H_Mou_inv, C=0)[:, :, 0]
            # pseudo_dab = hema_texture * mask[0, :, :, 1]
            # res[i * net_sizein:(i + 1) * net_sizein,
            # j * net_sizein:(j + 1) * net_sizein,
            # -1] = pseudo_dab
            # res[i * net_sizein:(i + 1) * net_sizein,
            # j * net_sizein:(j + 1) * net_sizein,
            # 0] = hema_texture * 0.5
    iou /= (w_num * h_num)
    f1 /= w_num * h_num
    print(iou, f1)
    res = hecconv(res, H_ki67)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(res)
    plt.axis("off")
    fig.tight_layout()
    plt.show()
