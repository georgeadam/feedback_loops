from math import ceil
import numpy as np

def ece(y, y_pred, y_prob, n_bins=10):
    if len(y_prob.shape) > 1:
        y_prob = y_prob[np.arange(len(y_pred)), y_pred]

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    y_cor = (y_pred == y).astype(int)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_cor = np.bincount(binids, weights=y_cor, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    acc = bin_cor / bin_total
    conf = bin_sums / bin_total

    ece = 0.0

    for i in range(len(bins)):
        if np.isnan(acc[i]):
            continue

        diff = np.abs(acc[i] - conf[i])
        ece += bin_total[i] * diff

    ece /= np.sum(bin_total)

    return ece


def mce(y, y_pred, y_prob, n_bins=10):
    if len(y_prob.shape) > 1:
        y_prob = y_prob[np.arange(len(y_pred)), y_pred]

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    y_cor = (y_pred == y).astype(int)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_cor = np.bincount(binids, weights=y_cor, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    acc = bin_cor / bin_total
    conf = bin_sums / bin_total

    max_diff = 0

    for i in range(len(bins)):
        if np.isnan(acc[i]):
            continue

        diff = np.abs(acc[i] - conf[i])

        if diff > max_diff:
            max_diff = diff

    return max_diff