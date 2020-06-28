from numba import jit
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from typing import Dict, Tuple, SupportsFloat

@jit
def fast_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == np.sum(y_true):
        return 0.0
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


def eval_model(y: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)

    samples = float(len(y_pred))

    return tn / samples, fp / samples, fn / samples, tp / samples


def confusion_matrix_custom(y: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tn = int(np.sum(np.logical_and(y_pred == 0, y == 0)))
    tp = int(np.sum(np.logical_and(y_pred == 1, y == 1)))

    fp = int(np.sum(np.logical_and(y_pred == 1, y == 0)))
    fn = int(np.sum(np.logical_and(y_pred == 0, y == 1)))

    return tn, fp, fn, tp


def precision_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)

    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)


def recall_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)

    return tp / (tp + fn)


def f1_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)

    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except:
        f1 =  0

    return f1


def compute_all_rates(y: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, initial: bool=False) -> Dict[str, float]:
    samples = float(len(y))
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)
    tnr, fpr, fnr, tpr = tn / (tn + fp), fp / (fp + tn), fn / (tp + fn), tp / (tp + fn)

    precision = precision_score(y, y_pred)
    if y_prob.shape[1] > 1:
        auc = fast_auc(y, y_prob[:, 1])
    else:
        auc = fast_auc(y, y_prob[:, 0])

    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    aupr = average_precision_score(y, y_pred)

    fp_idx = np.logical_and(y == 0, y_pred == 1)
    pos_idx = y == 1

    acc = float(np.sum(y == y_pred) / samples)

    if y_prob.shape[1] > 1:
        fp_conf = np.mean(y_prob[fp_idx, 1])
        pos_conf = np.mean(y_prob[pos_idx, 1])
    else:
        fp_conf = np.mean(y_prob[fp_idx, 0])
        pos_conf = np.mean(y_prob[pos_idx, 0])

    if initial:
        fp_count = 0
        total_samples = 0
    else:
        fp_count = int(np.sum(fp_idx))
        total_samples = len(y)

    if initial:
        rates = {"tnr": tnr, "fpr": fpr, "fnr": fnr, "tpr": tpr, "precision": precision, "recall": recall, "f1": f1,
                 "auc": auc, "aupr": aupr, "loss": None, "fp_conf": fp_conf, "pos_conf": pos_conf, "fp_count": fp_count,
                 "total_samples": total_samples, "fp_prop": 0.0, "acc": acc, "detection": None}
    else:
        rates = {"tnr": tnr, "fpr": fpr, "fnr": fnr, "tpr": tpr, "precision": precision, "recall": recall, "f1": f1,
                 "auc": auc, "aupr": aupr, "loss": None, "fp_conf": fp_conf, "pos_conf": pos_conf, "fp_count": fp_count,
                 "total_samples": total_samples, "acc": acc, "detection": None}

    return rates


def compute_fp_portion(y: np.ndarray, y_pred: np.ndarray) -> float:
    fp_idx = np.logical_and(y == 0, y_pred == 1)

    return np.sum(fp_idx) / len(y)