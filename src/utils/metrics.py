from numba import jit
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from typing import List, Dict, Tuple, SupportsFloat

from src.utils.detection import detect_feedback_loop
from src.utils.typing import Model, Transformer


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

    fp_count = int(np.sum(fp_idx))
    total_samples = len(y)

    rates = {"tnr": tnr, "fpr": fpr, "fnr": fnr, "tpr": tpr, "precision": precision, "recall": recall, "f1": f1,
             "auc": auc, "aupr": aupr, "loss": None, "fp_conf": fp_conf, "pos_conf": pos_conf, "fp_count": fp_count,
             "total_samples": total_samples, "acc": acc, "detection": None}

    return rates


def compute_rl_rates(y: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)
    tnr, fpr, fnr, tpr = tn / (tn + fp), fp / (fp + tn), fn / (tp + fn), tp / (tp + fn)

    f1 = f1_score(y, y_pred)

    rates = {"tnr": tnr, "fpr": fpr, "fnr": fnr, "tpr": tpr, "f1": f1}

    return rates


def fpr(y: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix_custom(y, y_pred)
    fpr = fp / (fp + tn)

    return fpr


def compute_model_fpr(model: Model, x: np.ndarray, y: np.ndarray, scaler: Transformer):
    y_prob = model.predict_proba(scaler.transform(x))
    y_pred = y_prob[:, 1] > model.threshold

    fp_idx = np.logical_and(y == 0, y_pred == 1)
    neg = np.sum(y == 0)

    return float(np.sum(fp_idx) / neg)


class RateTracker():
    def __init__(self):
        self._rates = {}

    def get_rates(self):
        return self._rates

    def update_rates(self, y, pred, prob):
        new_rates = compute_all_rates(y, pred, prob)

        for key, value in new_rates.items():
            if key in self._rates.keys():
                self._rates[key].append(value)
            else:
                self._rates[key] = [value]

    def update_detection(self, y_train, y_update):
        p = detect_feedback_loop(y_train, y_update)

        self._rates["detection"][-1] = p


class PredictionTracker():
    def __init__(self):
        self._predictions = {"y": [],"prob": [], "pred": [], "label": [], "update_num": [], "threshold": []}

    def get_predictions(self):
        return self._predictions

    def update_predictions(self, y, pred, prob, update_num, threshold):
        self._predictions["prob"] += list(prob)
        self._predictions["pred"] += list(pred)
        self._predictions["y"] += list(y)
        self._predictions["update_num"] += [update_num] * len(y)
        self._predictions["threshold"] += [threshold] * len(y)


def create_empty_rates() -> Dict[str, List]:
    return {"fpr": [], "tpr": [], "fnr": [], "tnr": [], "precision": [], "recall": [], "f1": [], "auc": [],
            "loss": [], "aupr": [], "fp_conf": [], "pos_conf": [], "fp_count": [], "total_samples": [],
            "fp_prop": [], "acc": [], "detection": [], "seed": []}


def create_empty_predictions() -> Dict[str, List]:
    return {"y": [],"prob": [], "pred": [], "label": [], "update_num": [], "threshold": [], "seed": []}