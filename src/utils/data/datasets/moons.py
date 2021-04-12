import numbers
import numpy as np

from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle

from typing import Callable, Tuple
from .utils import corrupt_labels


def make_moons(n_samples=100, *, start=0.0, end=np.pi, shuffle=True, noise=None, random_state=None):
    """Make two interleaving half circles.
       A simple toy dataset to visualize clustering and classification
       algorithms. Read more in the :ref:`User Guide <sample_generators>`.
       Parameters
       ----------
       n_samples : int or tuple of shape (2,), dtype=int, default=100
           If int, the total number of points generated.
           If two-element tuple, number of points in each of two moons.
           .. versionchanged:: 0.23
              Added two-element tuple.
       shuffle : bool, default=True
           Whether to shuffle the samples.
       noise : float, default=None
           Standard deviation of Gaussian noise added to the data.
       random_state : int, RandomState instance or None, default=None
           Determines random number generation for dataset shuffling and noise.
           Pass an int for reproducible output across multiple function calls.
           See :term:`Glossary <random_state>`.
       Returns
       -------
       X : ndarray of shape (n_samples, 2)
           The generated samples.
       y : ndarray of shape (n_samples,)
           The integer labels (0 or 1) for class membership of each sample.
       """

    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError('`n_samples` can be either an int or '
                             'a two-element tuple.') from e

    generator = check_random_state(random_state)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(start, end, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(start, end, n_samples_in)) - .5

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.ones(n_samples_out, dtype=np.intp),
                   np.zeros(n_samples_in, dtype=np.intp)])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


def generate_moons_dataset(start: float=0.0, end: float=np.pi, noise: float=0.0) -> Callable:
    def wrapped(n_train: int, n_val: int, n_update: int, n_test: int, num_features: int=2) -> Tuple:
        x_train, y_train = make_moons(n_train, start=0.0, end=np.pi, noise=0.1)
        x_val, y_val = make_moons(n_val, start=0.0, end=np.pi, noise=0.1)
        x_update, y_update = make_moons(n_update, start=start, end=end, noise=0.1)
        x_test, y_test = make_moons(n_test, start=0.0, end=end, noise=0.0)

        cols = np.arange(x_train.shape[1])

        if noise > 0.0:
            y_train = corrupt_labels(y_train, noise)

        data = {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_update": x_update,
                "y_update": y_update, "x_test": x_test, "y_test": y_test}

        return data, cols

    return wrapped


def generate_moons_noisy_update_dataset(start: float=0.0, end: float=np.pi, noise: float=0.0) -> Callable:
    def wrapped(n_train: int, n_val: int, n_update: int, n_test: int, num_features: int=2) -> Tuple:
        x_train, y_train = make_moons(n_train, start=0.0, end=np.pi, noise=0.1)
        x_val, y_val = make_moons(n_val, start=0.0, end=np.pi, noise=0.1)
        x_update, y_update = make_moons(n_update, start=start, end=end, noise=0.1)
        x_test, y_test = make_moons(n_test, start=0.0, end=end, noise=0.0)

        cols = np.arange(x_train.shape[1])

        if noise > 0.0:
            y_train = corrupt_labels(y_train, noise)
            y_val = corrupt_labels(y_val, noise)
            y_update = corrupt_labels(y_update, noise)

        data = {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val, "x_update": x_update,
                "y_update": y_update, "x_test": x_test, "y_test": y_test}

        return data, cols

    return wrapped