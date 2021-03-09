import copy
import numpy as np

from typing import Any, Callable, Tuple
from sklearn.model_selection import train_test_split


def generate_real_dataset_static(fn: Callable, path: str=None, balanced: bool=False,
                                 categorical: bool=False, model: str=None):
    data = fn(path, categorical, model)

    year_idx = None
    normalize_cols = []

    for i, column in enumerate(data["X"].columns):
        if column == "year":
            year_idx = i
        elif not (column == "age" or column == "gender" or column == "ethnicity" or column.startswith("age_") or column.startswith("gender_")
            or column.startswith("ethnicity_")):
            # This assumes that the year column is the first column in the dataframe
                normalize_cols.append(i - 1)

    x = data["X"].to_numpy()
    y = data["y"].to_numpy()

    nan_idx = np.where(np.isnan(x))[0]
    x = np.delete(x, nan_idx, 0)
    y = np.delete(y, nan_idx, 0)

    if year_idx is not None:
        x = np.delete(x, year_idx, 1)

    def wrapped(n_train: float, n_val: float, n_update: float, n_test: float, *args: Any, **kwargs: Any) -> Tuple:
        x_copy = copy.deepcopy(x)
        y_copy = copy.deepcopy(y)

        if balanced:
            neg_idx = np.where(y_copy == 0)[0]
            pos_idx = np.where(y_copy == 1)[0]

            drop = len(neg_idx) - len(pos_idx)

            del_idx = np.random.choice(neg_idx, drop, replace=False)

            x_copy, y_copy = np.delete(x_copy, del_idx, 0), np.delete(y_copy, del_idx, 0)

        if n_test is None:
            n_test = len(y_copy) - (n_train + n_val + n_update)
        elif n_train < 1 or n_update < 1 or n_test < 1:
            n_train = int(len(y_copy) * n_train)
            n_val = int(len(y_copy) * n_val)
            n_update = int(len(y_copy) * n_update)
            n_test = int(len(y_copy) * n_test)

        x_train, x_test, y_train, y_test = train_test_split(x_copy, y_copy, test_size=n_val + n_update + n_test,
                                                            stratify=y_copy)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=n_update + n_test, stratify=y_test)
        x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test, stratify=y_test)

        data = {"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val,
                "x_update": x_update, "y_update": y_update, "x_test": x_test, "y_test": y_test}

        return data, normalize_cols

    return wrapped