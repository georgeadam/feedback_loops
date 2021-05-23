import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from typing import Callable
from .utils import corrupt_labels


def generate_blobs_dataset(noise: float=0.0) -> Callable:
    def wrapped(n_train: int, n_val: int, n_update: int, n_test: int, num_features: int=2):
        angle = np.random.rand(1)[0] * 2
        r = np.sqrt(3)

        cluster_1 = np.array([r * np.cos(angle), r * np.sin(angle)])
        cluster_2 = - cluster_1
        centers = np.array([cluster_1, cluster_2])

        x, y = make_blobs(n_train + n_val + n_update + n_test, num_features, centers)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_val + n_update + n_test)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = n_update + n_test)
        x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

        cols = np.arange(num_features)

        if noise > 0.0:
            y_train_corrupt = corrupt_labels(y_train, noise)
        else:
            y_train_corrupt = y_train

        data = {"x_train": x_train, "y_train": y_train_corrupt, "y_train_clean": y_train, "x_val": x_val, "y_val": y_val,
                "x_update": x_update, "y_update": y_update, "x_test": x_test, "y_test": y_test}

        return data, cols

    return wrapped
