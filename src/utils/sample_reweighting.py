import copy

import numpy as np


def get_weights(weight_type, sub_conf, sub_y, sub_y_unmodified, threshold):
    if weight_type == "confidence":
        sub_idx = sub_y == 0
        sub_weights = copy.deepcopy(sub_conf)
        sub_weights[sub_idx] = 1
        weights = sub_weights
    elif weight_type == "drop_low_confidence":
        sorted_idx = np.argsort(sub_conf)
        unsorted_idx = np.argsort(sorted_idx)
        sorted_y = sub_y[sorted_idx]
        sorted_pos_idx = np.where(sorted_y == 1)[0]
        sorted_pos_idx = sorted_pos_idx[: int(0.66 * len(sorted_pos_idx))]

        temp_idx = unsorted_idx[sorted_pos_idx]
        pos_idx = np.where(sub_conf[temp_idx] > threshold)[0]
        pos_idx = temp_idx[pos_idx]
        neg_idx = np.delete(np.arange(len(sub_y)), pos_idx)
        sub_weights = copy.deepcopy(sub_conf)
        sub_weights[neg_idx] = 1
        sub_weights[pos_idx] = 0

        weights = sub_weights
    elif weight_type == "oracle":
        fp_idx = np.logical_and(sub_conf > threshold, sub_y_unmodified == 0)
        sub_weights = np.ones(len(sub_y_unmodified))
        sub_weights[fp_idx] = 0

        weights = sub_weights
    elif weight_type == "random":
        pos_idx = np.where(sub_conf > threshold)[0]
        pos_percentage = (np.sum(sub_y == 1) / len(sub_y))
        drop_percentage = np.max((pos_percentage - 0.1) / pos_percentage, 0)
        drop_idx = np.random.choice(pos_idx, size=int(drop_percentage * len(pos_idx)), replace=False)
        sub_weights = np.ones(len(sub_y))
        sub_weights[drop_idx] = 0

        weights = sub_weights
    elif weight_type == "drop_everything":
        drop_idx = np.where(sub_conf > threshold)[0]
        sub_weights = np.ones(len(sub_y))
        sub_weights[drop_idx] = 0

        weights = sub_weights
    else:
        weights = np.array(np.ones(len(sub_y)))

    return weights