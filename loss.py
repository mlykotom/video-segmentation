import numpy as np

from metrics import dice_coef

smooth = 1e-5 # TODO is also in metrics.py


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def np_dice_coef(y_true, y_pred):
    tr = y_true.flatten()
    pr = y_pred.flatten()

    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)
