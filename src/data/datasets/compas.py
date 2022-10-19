import warnings

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .utils import corrupt_labels


def generate_compas_dataset(noise):
    def wrapped(n_train: float, n_val: float, n_update: float, n_test: float, *args):
        with warnings.catch_warnings():
            compas = fetch_openml("compas-two-years", as_frame=True, return_X_y=True)

        x, y = compas
        x["y"] = y
        x = x.dropna()
        y = x["y"]
        x = x.drop(["y"], axis=1)

        cat_type = CategoricalDtype(categories=[0.0, 1.0], ordered=True)

        x["c_charge_degree_F"] = x["c_charge_degree_F"].astype(cat_type)
        x["c_charge_degree_M"] = x["c_charge_degree_M"].astype(cat_type)

        new_cols = {}

        for prefix in ["age_cat_", "race_", "c_charge_degree_"]:
            cols = get_cols_with_prefix(x, prefix)
            new_col = pd.Series(["missing"] * len(x))

            for col in cols:
                value = col.split(prefix)[1]
                idx = np.where(x[col].cat.codes == 1)[0]
                new_col[idx] = value

            new_cols[prefix[:-1]] = new_col

        for prefix in ["age_cat_", "race_", "c_charge_degree_"]:
            cols = get_cols_with_prefix(x, prefix)

            for col in cols:
                x = x.drop(columns=col)

        for c, col in new_cols.items():
            x[c] = col

        dummy_x = pd.get_dummies(x)
        dummy_cols = dummy_x.select_dtypes(exclude=["float", "float32", "float64"]).columns

        numeric_cols = dummy_x.columns.difference(dummy_cols)
        numeric_col_indices = []

        for numeric_col in numeric_cols:
            index = dummy_x.columns.get_loc(numeric_col)
            numeric_col_indices.append(index)

        x_df = dummy_x
        x = dummy_x.to_numpy()
        y = pd.factorize(y, sort=True)[0]

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

        data = {"x_df": x_df, "x_train": x_train, "y_train_clean": y_train, "y_train": y_train_corrupt,
                "x_val": x_val, "y_val": y_val, "x_update": x_update, "y_update": y_update,
                "x_test": x_test, "y_test": y_test}

        return data, numeric_col_indices

    return wrapped


def get_cols_with_prefix(df, prefix):
    cols = []

    for col in df.columns:
        if col.startswith(prefix):
            cols.append(col)

    return cols
