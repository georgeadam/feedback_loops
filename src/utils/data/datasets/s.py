import numpy as np
from sklearn.model_selection import train_test_split

from typing import Tuple


def generate_s_shape_dataset(n_train: int, n_update: int, n_test: int, num_features: int=2, noise: float=0.0) -> Tuple:
    total_samples = (n_train + n_update + n_test) // 2
    y = np.linspace(2.0, 3.0, total_samples)
    x_0 = 0.25 * ((np.cos(3.5 * y) + np.random.normal(0, 0.1, len(y))) + 4) + 1.9
    x_1 = 0.25 * ((np.cos(3.5 * y) + np.random.normal(0, 0.1, len(y))) + 4.5) + 1.9

    x = np.concatenate([np.stack([x_0, y], 1),
                        np.stack([x_1, y], 1)])
    y = np.concatenate([np.zeros(total_samples),
                       np.ones(total_samples)])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_update + n_test)
    x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

    return x_train, y_train, x_update, y_update, x_test, y_test


def generate_long_s_shape_dataset(n_train: int, n_update: int, n_test: int, num_features: int=2, noise: float=0.0) -> Tuple:
    total_samples = (n_train + n_update + n_test) // 2
    y = np.linspace(2.0, 5.0, total_samples)
    x_0 = 0.25 * ((np.cos(3.5 * y) + np.random.normal(0, 0.1, len(y))) + 4) + 1.9
    x_1 = 0.25 * ((np.cos(3.5 * y) + np.random.normal(0, 0.1, len(y))) + 4.5) + 1.9

    x = np.concatenate([np.stack([x_0, y], 1),
                        np.stack([x_1, y], 1)])
    y = np.concatenate([np.zeros(total_samples),
                       np.ones(total_samples)])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_update + n_test)
    x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

    return x_train, y_train, x_update, y_update, x_test, y_test




