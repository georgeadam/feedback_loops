import numpy as np

from typing import Callable, Tuple


def make_gaussian_data(m0: float, m1: float, s0: float, s1: float, n: int, p0: float, p1: float, num_features: int=2,
                       noise: float=0.0) -> Tuple[np.ndarray, np.ndarray]:
    neg_samples = np.random.multivariate_normal(m0 * np.ones(num_features), s0 * np.eye(num_features), int(n * p0))
    pos_samples = np.random.multivariate_normal(m1 * np.ones(num_features), s1 * np.eye(num_features), int(n * p1))

    x = np.concatenate((neg_samples, pos_samples))
    y = np.concatenate((np.zeros(len(neg_samples)), np.ones(len(pos_samples))))

    if noise > 0.0:
        neg_idx = y == 0
        bernoulli = np.random.choice([False, True], len(y), p=[1 - noise, noise])
        neg_idx = np.logical_and(neg_idx, bernoulli)
        neg_idx = np.where(neg_idx)
        y[neg_idx] = 1

    idx = np.random.choice(len(x), len(x), replace=False)
    x = x[idx]
    y = y[idx]

    return x, y


def generate_gaussian_dataset(m0: float=-1, m1: float=1, s0: float=1, s1: float=1, p0: float=0.5, p1: float=0.5) -> Callable:
    def wrapped(n_train: int, n_update: int, n_test: int, num_features: int=2, noise: float=0.0):
        x_train, y_train = make_gaussian_data(m0, m1, s0, s1, n_train, p0, p1, num_features=num_features, noise=noise)

        x_update, y_update = make_gaussian_data(m0, m1, s0, s1, n_update, p0, p1, num_features=num_features)
        x_test, y_test = make_gaussian_data(m0, m1, s0, s1, n_test, p0, p1, num_features=num_features)

        return x_train, y_train, x_update, y_update, x_test, y_test

    return wrapped
