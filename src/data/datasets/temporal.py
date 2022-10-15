import copy
import numpy as np

from sklearn.model_selection import train_test_split

from typing import Any, Callable, Tuple


def generate_real_dataset_temporal(fn: Callable, path: str=None, balanced: bool=False, categorical: bool=False,
                                   model: str=None, tyl: int=None) -> Callable:
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

    if "year" in data["X"].columns:
        unique_years = np.unique(data["X"]["year"])
        years = data["X"]["year"].to_numpy()
    else:
        unique_years = None
        years = None

    x_df = data["X"]
    y_df = data["X"]

    x = data["X"].to_numpy()
    y = data["y"].to_numpy()

    nan_idx = np.where(np.isnan(x))[0]
    x = np.delete(x, nan_idx, 0)
    y = np.delete(y, nan_idx, 0)
    years = np.delete(years, nan_idx, 0)

    x_df = x_df.drop(nan_idx)
    x_df = x_df.drop(columns=["year"])
    y_df = y_df.drop(nan_idx)

    if year_idx is not None:
        x = np.delete(x, year_idx, 1)

    def wrapped(n_train: float, n_val: float, n_update: float, n_test: float, *args: Any, **kwargs: Any) -> Tuple:
        x_copy = copy.deepcopy(x)
        y_copy = copy.deepcopy(y)
        years_copy = copy.deepcopy(years)

        if balanced:
            del_idx = np.array([])

            for year in unique_years:
                idx = years == year
                neg_idx = y_copy == 0
                pos_idx = y_copy == 1

                neg_idx = np.logical_and(idx, neg_idx)
                pos_idx = np.logical_and(idx, pos_idx)

                drop = np.sum(neg_idx) - np.sum(pos_idx)
                neg_idx = np.where(neg_idx)[0]
                temp_idx = np.random.choice(neg_idx, drop, replace=False)

                del_idx = np.concatenate([del_idx, temp_idx])

            x_copy, y_copy = np.delete(x_copy, del_idx, 0), np.delete(y_copy, del_idx, 0)
            years_copy = np.delete(years_copy, del_idx, 0)

        train_idx = years_copy <= tyl
        x_train, y_train = x_copy[train_idx], y_copy[train_idx]
        x_rest, y_rest = x_copy[~train_idx], y_copy[~train_idx]
        years_rest = years_copy[~train_idx]
        n_val = int(len(x_train) * n_val)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=n_val, stratify=y_train)

        data = {"x_df": x_df, "x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val,
                "x_rest": x_rest, "y_rest": y_rest, "years_rest": years_rest}

        return data, normalize_cols

    return wrapped


def generate_image_dataset_temporal(fn: Callable, balanced: bool=False, tyl: int=None):
    data = fn()

    train_indices = np.where(data.csv["StudyDate_DICOM"] % 100000 < tyl)
    rest_indices = np.where(data.csv["StudyDate_DICOM"] >= tyl)

    train_data = copy.deepcopy(data)
    rest_data = copy.deepcopy(data)

    train_data.csv = train_data.csv.iloc[train_indices]
    train_data.labels = train_data.labels.iloc[train_indices]

    rest_data.csv = rest_data.csv.iloc[rest_indices]
    rest_data.labels = rest_data.labels.iloc[rest_indices]

    return
