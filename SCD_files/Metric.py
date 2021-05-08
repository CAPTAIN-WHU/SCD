from PIL import Image
import numpy as np
import math
import os


num_class = 37
IMAGE_FORMAT = '.png'
INFER_DIR = './prediction_dir/'
LABEL_DIR = './label_dir/'


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def Eval():
    name_list = sorted(os.listdir(INFER_DIR))
    hist = np.zeros((num_class, num_class))
    for idx in range(len(name_list)):
        name = name_list[idx].split('.')[0]
        infer_file = INFER_DIR + '/' + str(name) + IMAGE_FORMAT
        label_file = LABEL_DIR + '/' + str(name) + IMAGE_FORMAT
        infer = Image.open(infer_file)
        label = Image.open(label_file)
        infer_array = np.array(infer)
        label_array = np.array(label)
        hist += get_hist(infer_array, label_array)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    print('Mean IoU = %.5f' % IoU_mean)
    print('Sek = %.5f' % Sek)


if __name__ == '__main__':
    Eval()
