from numba import jit
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_model(y, y_pred):
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)

    samples = float(len(y_pred))

    return tn / samples, fp / samples, fn / samples, tp / samples


def confusion_matrix_custom(y, y_pred):
    tn = np.sum(np.logical_and(y_pred == 0, y == 0))
    tp = np.sum(np.logical_and(y_pred == 1, y == 1))

    fp = np.sum(np.logical_and(y_pred == 1, y == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y == 1))

    return tn, fp, fn, tp


def precision_score(y, y_pred):
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)

    return tp / (tp + fp)


def recall_score(y, y_pred):
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)

    return tp / (tp + fn)


def f1_score(y, y_pred):
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)

    return (2 * precision * recall) / (precision + recall)


def compute_all_rates(y, y_pred, y_prob):
    samples = float(len(y))
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)
    tnr, fpr, fnr, tpr = tn / samples, fp / samples, fn / samples, tp / samples

    precision = precision_score(y, y_pred)
    auc = fast_auc(y, y_prob[:, 1])
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    aupr = average_precision_score(y, y_pred)

    fp_idx = np.logical_and(y == 0, y_pred == 1)
    pos_idx = y == 1

    fp_conf = np.mean(y_prob[fp_idx, 1])
    pos_conf = np.mean(y_prob[pos_idx, 1])

    rates = {"tnr": tnr, "fpr": fpr, "fnr": fnr, "tpr": tpr, "precision": precision, "recall": recall, "f1": f1,
             "auc": auc, "aupr": aupr, "loss": None, "fp_conf": fp_conf, "pos_conf": pos_conf}

    return rates