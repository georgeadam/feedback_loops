import copy
import numpy as np


def corrupt_labels(y, noise):
    y = copy.deepcopy(y)
    flip_indices = np.random.choice(len(y), int(noise * len(y)), replace=False)
    y[flip_indices] = 1 - y[flip_indices]

    return y


def perturb_labels_fp(y: np.ndarray, rate: float=0.05) -> np.ndarray:
    y_copy = copy.deepcopy(y)
    n_pert = int(len(y_copy) * rate)

    neg_idx = y_copy == 0
    neg_idx = np.where(neg_idx)[0]

    pert_idx = np.random.choice(neg_idx, n_pert, replace=False)

    y_copy[pert_idx] = 1

    return y_copy