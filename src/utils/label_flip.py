import copy

import numpy as np


def flip_labels(flip_type, sub_conf, sub_y, sub_y_unmodified, threshold):
    if flip_type == "flip_low_confidence":
        sorted_idx = np.argsort(sub_conf)
        unsorted_idx = np.argsort(sorted_idx)
        sorted_y = sub_y[sorted_idx]
        sorted_pos_idx = np.where(sorted_y == 1)[0]
        sorted_pos_idx = sorted_pos_idx[: int(0.66 * len(sorted_pos_idx))]

        temp_idx = unsorted_idx[sorted_pos_idx]
        pos_idx = np.where(sub_conf[temp_idx] > threshold)[0]
        pos_idx = temp_idx[pos_idx]
        new_y = copy.deepcopy(sub_y)
        new_y[pos_idx] = 0
    elif flip_type == "oracle":
        fp_idx = np.logical_and(sub_conf > threshold, sub_y_unmodified == 0)
        new_y = copy.deepcopy(sub_y)
        new_y[fp_idx] = 0
    elif flip_type == "random":
        pos_idx = np.where(sub_conf > threshold)[0]
        pos_percentage = (np.sum(sub_y == 1) / len(sub_y))
        flip_percentage = np.max((pos_percentage - 0.1) / pos_percentage, 0)
        flip_idx = np.random.choice(pos_idx, size=int(flip_percentage * len(pos_idx)), replace=False)
        new_y = copy.deepcopy(sub_y)
        new_y[flip_idx] = 0
    elif flip_type == "flip_everything":
        flip_idx = np.where(sub_conf > threshold)[0]
        new_y = copy.deepcopy(sub_y)
        new_y[flip_idx] = 0
    else:
        new_y = sub_y

    return new_y