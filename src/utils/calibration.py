from math import ceil
import numpy as np

def ece(y, y_pred, y_prob, bins=10):
    y_prob = y_prob[np.arange(len(y_pred)), y_pred]
    idx = np.argsort(y_prob)

    y = y[idx]
    y_pred = y_pred[idx]
    y_prob = y_prob[idx]

    bin_size = ceil(len(y) / bins)

    ece = 0.0

    for i in range(bins):
        temp_y = y[i * bin_size: (i + 1) * bin_size]
        temp_y_pred = y_pred[i * bin_size: (i + 1) * bin_size]
        temp_y_prob = y_prob[i * bin_size: (i + 1) * bin_size]

        acc = np.sum(temp_y == temp_y_pred) / len(temp_y)
        confidence = np.mean(temp_y_prob)

        diff = np.abs(acc - confidence)

        ece += len(temp_y) * diff

    ece /= len(y)

    return ece
