import copy
import numbers
import numpy as np
import os
import pandas as pd
from scipy.stats import multivariate_normal

from sklearn.datasets import make_classification, make_gaussian_quantiles, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import _deprecate_positional_args
from src.utils.typing import DataFn

from omegaconf import DictConfig
from settings import ROOT_DIR
from typing import Any, Callable, Dict, List, Tuple



STATIC_DATA_TYPES = ["gaussian", "sklearn", "moons", "mimic_iii", "support2"]
TEMPORAL_DATA_TYPES = ["mimic_iv", "mimic_iv_12h", "mimic_iv_24h"]

mimic_iv_paths = {"mimic_iv": {"path": "mimic_iv_datasets_with_year_imputed.csv", "categorical": False},
                  "mimic_iv_12h": {"path": "mimic_iv_datasets_with_year_12hrs_imputed.csv", "categorical": False},
                  "mimic_iv_24h": {"path": "mimic_iv_datasets_with_year_24hrs_imputed.csv", "categorical": False},
                  "mimic_iv_demographic": {"path": "mimic_iv_datasets_with_year_48hrs_imputed_age_race_gender.csv", "categorical": True},
                  "mimic_iv_12h_demographic": {"path": "mimic_iv_datasets_with_year_12hrs_imputed_age_race_gender.csv", "categorical": True},
                  "mimic_iv_24h_demographic": {"path": "mimic_iv_datasets_with_year_24hrs_imputed_age_race_gender.csv", "categorical": True}}

def get_data_fn(d: DictConfig, m: DictConfig) -> DataFn:
    if d.type == "gaussian":
        if hasattr(d, "m0"):
            return generate_gaussian_dataset(d.m0, d.m1, d.s0, d.s1, d.p0, d.p1)
        else:
            return generate_gaussian_dataset()
    elif d.type == "sklearn":
        return generate_sklearn_make_classification_dataset(d.noise)
    elif d.type == "mimic_iii":
        return generate_real_dataset(load_mimic_iii_data, balanced=d.balanced)
    elif "mimic_iv" in d.type:
        return generate_real_dataset(load_mimic_iv_data,
                                     mimic_iv_paths[d.type]["path"], d.balanced, temporal=d.temporal,
                                     categorical=mimic_iv_paths[d.type]["categorical"], model=m.type)
    elif d.type == "moons":
        return generate_moons_dataset(d.start, d.end, d.noise)
    elif d.type == "circles":
        return generate_circles_dataset(d.noise)
    elif d.type == "support2":
        return generate_real_dataset(load_support2cls_data)


def perturb_labels_fp(y: np.ndarray, rate: float=0.05) -> np.ndarray:
    y_copy = copy.deepcopy(y)
    n_pert = int(len(y_copy) * rate)

    neg_idx = y_copy == 0
    neg_idx = np.where(neg_idx)[0]

    pert_idx = np.random.choice(neg_idx, n_pert, replace=False)

    y_copy[pert_idx] = 1

    return y_copy


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


def make_trend_gaussian_data(m0: float, m1: float, s0: float, s1: float, n: int, num_features: int=2, noise: float=0.0,
                             uniform_range: List[float]=[-3, 3]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(uniform_range[0], uniform_range[1], (n, num_features))

    pdf0 = multivariate_normal(mean=np.ones(num_features) * m0, cov=np.eye(num_features) * s0).pdf(x)
    pdf1 = multivariate_normal(mean=np.ones(num_features) * m1, cov=np.eye(num_features) * s1).pdf(x)

    p_y = pdf1 / (pdf0 + pdf1)

    y = []

    for i in range(len(x)):
        y.append(np.random.choice([0, 1], p=[1 - p_y[i], p_y[i]]))

    y = np.array(y)
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


def generate_sklearn_make_classification_dataset(noise: float=0.0) -> Callable:
    def wrapped(n_train: int, n_update: int, n_test: int, num_features: int=2):
        x, y = make_classification(n_train + n_update + n_test, n_informative=num_features, n_features=num_features,
                                   n_classes=2, n_clusters_per_class=2, n_redundant=0, flip_y=0, class_sep=1.0)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_update + n_test)
        x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

        cols = np.arange(num_features)

        if noise > 0.0:
            neg_idx = y_train == 0
            bernoulli = np.random.choice([False, True], len(y_train), p=[1 - noise, noise])
            neg_idx = np.logical_and(neg_idx, bernoulli)
            neg_idx = np.where(neg_idx)
            y_train[neg_idx] = 1

        return x_train, y_train, x_update, y_update, x_test, y_test, cols

    return wrapped


def generate_moons_dataset(start: float=0.0, end: float=np.pi, noise: float=0.0) -> Callable:
    def wrapped(n_train: int, n_update: int, n_test: int, num_features: int=2) -> Tuple:
        x_train, y_train = make_moons(n_train, start=0.0, end=np.pi, noise=noise)
        x_update, y_update = make_moons(n_update, start=start, end=end, noise=noise)
        x_test, y_test = make_moons(n_test, start=0.0, end=end, noise=0.0)

        return x_train, y_train, x_update, y_update, x_test, y_test, None

    return wrapped


def generate_circles_dataset(noise: float=0.0):
    def wrapped(n_train: int, n_update: int, n_test: int, num_features: int):
        x_train, y_train = make_circles(n_train, noise=noise)
        x_update, y_update = make_circles(n_update, noise=noise)
        x_test, y_test = make_circles(n_test, noise=0.0)

        return x_train, y_train, x_update, y_update, x_test, y_test, None

    return wrapped


def generate_gaussian_quantile_dataset(n_train: int, n_update: int, n_test: int, num_features: int=2, noise: float=0.0) -> Tuple:
    X1, y1 = make_gaussian_quantiles(cov=2.,
                                     n_samples=int((n_train + n_update + n_test) / 2), n_features=2,
                                     n_classes=2)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                     n_samples=int((n_train + n_update + n_test) / 2), n_features=2,
                                     n_classes=2)
    x = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_update + n_test)
    x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

    return x_train, y_train, x_update, y_update, x_test, y_test


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


def load_mimic_iii_data(*args: Any, **kargs: Any) -> Dict[str, Any]:
    df_adult = pd.read_csv(os.path.join(ROOT_DIR, 'adult_icu.gz'), compression='gzip')

    train_cols = [
        'age', 'first_hosp_stay', 'first_icu_stay', 'eth_asian',
        'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
        'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN',
        'admType_URGENT', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
        'sysbp_min', 'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max',
        'diasbp_mean', 'meanbp_min', 'meanbp_max', 'meanbp_mean',
        'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min',
        'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean',
        'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin',
        'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
        'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 'phosphate',
        'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc']

    label = 'mort_icu'
    X_df = df_adult[train_cols]
    y_df = df_adult[label]

    dataset = {
        'problem': 'classification',
        'X': X_df,
        'y': y_df,
        'd_name': 'mimic_iii',
    }

    return dataset


def generate_real_dataset(fn: Callable, path: str=None, balanced: bool=False, temporal: bool=True, categorical: bool=False,
                          model: str=None) -> Callable:
    data = fn(path, categorical, model)

    year_idx = None
    normalize_cols = []

    for i, column in enumerate(data["X"].columns):
        if column == "year":
            year_idx = i
        elif not (column == "age" or column == "gender" or column == "ethnicity" or column.startswith("age_") or column.startswith("gender_")
            or column.startswith("ethnicity_")):
                normalize_cols.append(i - 1)

    if "year" in data["X"].columns and temporal:
        years = np.unique(data["X"]["year"])
    else:
        years = None

    x = data["X"].to_numpy()
    y = data["y"].to_numpy()

    nan_idx = np.where(np.isnan(x))[0]
    x = np.delete(x, nan_idx, 0)
    y = np.delete(y, nan_idx, 0)

    print("a")

    def wrapped(n_train: float, n_update: float, n_test: float, **kwargs: Any) -> Tuple:
        x_copy = copy.deepcopy(x)
        y_copy = copy.deepcopy(y)

        if balanced:
            if years is not None:
                del_idx = np.array([])

                for year in years:
                    idx = x_copy[:, 0] == year
                    neg_idx = y_copy == 0
                    pos_idx = y_copy == 1

                    neg_idx = np.logical_and(idx, neg_idx)
                    pos_idx = np.logical_and(idx, pos_idx)

                    drop = np.sum(neg_idx) - np.sum(pos_idx)
                    neg_idx = np.where(neg_idx)[0]
                    temp_idx = np.random.choice(neg_idx, drop, replace=False)

                    del_idx = np.concatenate([del_idx, temp_idx])

                x_copy, y_copy = np.delete(x_copy, del_idx, 0), np.delete(y_copy, del_idx, 0)
            else:
                neg_idx = np.where(y_copy == 0)[0]
                pos_idx = np.where(y_copy == 1)[0]

                drop = len(neg_idx) - len(pos_idx)

                del_idx = np.random.choice(neg_idx, drop, replace=False)

                x_copy, y_copy = np.delete(x_copy, del_idx, 0), np.delete(y_copy, del_idx, 0)

        if n_test is None:
            n_test = len(y_copy) - (n_train + n_update)
        elif n_train < 1 or n_update < 1 or n_test < 1:
            n_train = int(len(y_copy) * n_train)
            n_update = int(len(y_copy) * n_update)
            n_test = int(len(y_copy) * n_test)

        if year_idx is not None and not temporal:
            x_copy = np.delete(x_copy, year_idx, 1)

        x_train, x_test, y_train, y_test = train_test_split(x_copy, y_copy, test_size=n_update + n_test,
                                                            stratify=y_copy)
        x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test, stratify=y_test)

        return x_train, y_train, x_update, y_update, x_test, y_test, normalize_cols

    return wrapped


def load_support2cls_data() -> Dict[str, Any]:
    df = pd.read_csv(os.path.join(ROOT_DIR, 'support2.csv'))
    one_hot_encode_cols = ['sex', 'dzclass', 'race', 'ca', 'income']
    target_variables = ['hospdead']
    remove_features = ['death', 'slos', 'd.time', 'dzgroup', 'charges', 'totcst',
                        'totmcst', 'aps', 'sps', 'surv2m', 'surv6m', 'prg2m',
                       'prg6m',
                       'dnr', 'dnrday', 'avtisst', 'sfdm2']
    df = df.drop(remove_features, axis=1)
    rest_colmns = [c for c in df.columns if c not in (one_hot_encode_cols + target_variables)]
    # Impute the missing values for 0.
    df[rest_colmns] = df[rest_colmns].fillna(0.)
    df = pd.get_dummies(df, prefix=one_hot_encode_cols)
    X_df = df.drop(target_variables, axis=1)
    X_df = X_df.astype('float64')

    y_df = df[target_variables[0]]
    dataset = {
        'problem': 'classification',
        'X': X_df,
        'y': y_df,
        'd_name': 'support2cls',
    }

    return dataset


def load_mimic_iv_data(path: str, categorical: bool=False, model: str="lr") -> Dict[str, Any]:
    df_adult = pd.read_csv(os.path.join(ROOT_DIR, path))

    train_cols = [
        'year',
        'HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean', 'SysBP_Min',
        'SysBP_Max', 'SysBP_Mean', 'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
        'MeanBP_Min', "MeanBP_Max", "MeanBP_Mean", "RespRate_Min", "RespRate_Max",
        "RespRate_Mean", "TempC_Min", "TempC_Max", "TempC_Mean", "SpO2_Min", "SpO2_Max",
        "SpO2_Mean", "Glucose_Min", "Glucose_Max", "Glucose_Mean", "ANIONGAP", "ALBUMIN",
        "BICARBONATE", "BILIRUBIN", "CREATININE", "CHLORIDE", "GLUCOSE", "HEMATOCRIT",
        "HEMOGLOBIN", "LACTATE", "MAGNESIUM", "PHOSPHATE", "PLATELET", "POTASSIUM", "PTT",
        "INR", "PT", "SODIUM", "BUN", "WBC"]
    if categorical:
        train_cols.append("age")

    categorical_cols = ["gender", "ethnicity"]

    if categorical:
        train_cols += categorical_cols

    label = 'mort_icu'
    X_df = df_adult[train_cols]

    if categorical and model != "random_forest":
        X_df = pd.get_dummies(X_df, columns=categorical_cols)
        X_df["age"] = X_df["age"].map({"0 - 10": 5, "10 - 20": 15, "20 - 30": 25, "30 - 40": 35,
                                       "40 - 50": 45, "50 - 60": 55, "60 - 70": 65, "70 - 80": 75, "> 80": 85})
    elif categorical:
        for col in categorical_cols:
            unique_vals = np.unique(X_df[col])
            replace = np.arange(len(unique_vals))
            mapping = {unique_vals[i]: replace[i] for i in range(len(unique_vals))}
            X_df[col] = X_df[col].map(mapping)

        X_df["age"] = X_df["age"].map({"0 - 10": 5, "10 - 20": 15, "20 - 30": 25, "30 - 40": 35,
                                       "40 - 50": 45, "50 - 60": 55, "60 - 70": 65, "70 - 80": 75, "> 80": 85})

    y_df = df_adult[label]

    dataset = {
        'problem': 'classification',
        'X': X_df,
        'y': y_df,
        'd_name': 'mimic_iv',
    }

    return dataset
