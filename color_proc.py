import matplotlib.pyplot as plt
from numpy import *
# from skimage.color import rgb2hed, hed2rgb, separate_stains, combine_stains
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D  # --- For 3D surface drawing

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
    #     from skimage.exposure import rescale_intensity
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

M_article = array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78]
])

# color maps for visualization

cmap_hema = LinearSegmentedColormap.from_list('cmap_hema', ['white', 'navy'])
cmap_eosin = LinearSegmentedColormap.from_list('cmap_eosin', ['white', 'darkviolet'])
cmap_dab = LinearSegmentedColormap.from_list('cmap_dab', ['white', 'saddlebrown'])


def single_prediction(im_in, net, net_sizein):
    W, H = im_in.shape[0], im_in.shape[1]
    if W % net_sizein != 0 or H % net_sizein != 0:
        raise ValueError("")
    # TODO: 增加对padding的支持，输入不一定是256的整数倍
    w_num, h_num = W // net_sizein, H // net_sizein
    res = zeros((W,H))
    for i in range(w_num):
        for j in range(h_num):
            chip = im_in[i * net_sizein:(i + 1) * net_sizein,
                   j * net_sizein:(j + 1) * net_sizein,
                   :]
            mask = net(chip)
            hema_texture = rgbdeconv(chip, M_article, C=0)[:, :, 0]
            pseudo_dab = hema_texture * mask
            res[i * net_sizein:(i + 1) * net_sizein,
            j * net_sizein:(j + 1) * net_sizein,
            :] = pseudo_dab
    plt.figure()
    plt.imshow(pseudo_dab, cmap=cmap_hema)
    plt.axis("off")
