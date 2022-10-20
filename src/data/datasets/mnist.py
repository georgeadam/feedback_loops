import numpy as np
import os
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .utils import corrupt_labels

from settings import ROOT_DIR
from scipy.io import arff

mnist = None

def generate_mnist_dataset(noise):
    def wrapped(n_train: float, n_val: float, n_update: float, n_test: float, *args):
        global mnist

        if mnist is None:
            mnist = arff.loadarff(os.path.join(ROOT_DIR, "data/mnist/mnist_784.arff"))[0]
            mnist = pd.DataFrame(mnist)

        features = [c for c in mnist.columns if not c.startswith("class")]
        x = mnist[features]
        y = mnist["class"].astype(int)

        x = x.to_numpy()
        y = y.to_numpy()

        x /= 255.0

        neg_indices = y == 1
        pos_indices = y == 7

        x = x[neg_indices | pos_indices]
        y[neg_indices] = 0
        y[pos_indices] = 1

        y = y[neg_indices | pos_indices]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_val + n_update + n_test)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=n_update + n_test)
        x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

        if noise.train > 0.0:
            y_train_corrupt = corrupt_labels(y_train, noise.train)
        else:
            y_train_corrupt = y_train

        if noise.val > 0.0:
            y_val = corrupt_labels(y_val, noise.val)

        if noise.future > 0.0:
            y_update = corrupt_labels(y_update, noise.future)

        data = {"x_train": x_train, "y_train_clean": y_train, "y_train": y_train_corrupt,
                "x_val": x_val, "y_val": y_val, "x_update": x_update, "y_update": y_update,
                "x_test": x_test, "y_test": y_test}

        return data, np.arange(784)

    return wrapped