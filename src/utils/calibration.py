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


def compute_bins_acc(y, y_prob, y_pred, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    y_cor = (y_pred == y).astype(int)
    tp = ((y_pred == y) & (y == 1)).astype(int)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_cor = np.bincount(binids, weights=y_cor, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    bin_tp = np.bincount(binids, weights=tp, minlength=len(bins))

    acc = bin_cor / bin_total
    conf = bin_sums / bin_total
    pos_prop_all = bin_true / np.sum(y)
    pos_prop_bin = bin_true / bin_total
    sens = bin_tp / bin_true

    return acc[:n_bins], conf[:n_bins], pos_prop_all[:n_bins], pos_prop_bin[:n_bins], sens[:n_bins]


def compute_bins_pos(y, y_prob, n_bins=10):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    prob_true = bin_true / bin_total
    prob_pred = bin_sums / bin_total

    return prob_true[:n_bins], prob_pred[:n_bins]