import copy
import numpy as np
import os
import pandas as pd
from scipy.stats import multivariate_normal

from sklearn.datasets import make_classification, make_gaussian_quantiles, make_moons
from sklearn.model_selection import train_test_split

from settings import ROOT_DIR


def get_data_fn(args):
    if args.data_type == "gaussian":
        return generate_gaussian_dataset(args.m0, args.m1, args.s0, args.s1, args.p0, args.p1)
    elif args.data_type == "sklearn":
        return generate_sklearn_make_classification_dataset
    elif args.data_type == "mimic":
        return generate_mimic_dataset()


def perturb_labels_fp(y, rate=0.05):
    y_copy = copy.deepcopy(y)
    n_pert = int(len(y_copy) * rate)

    neg_idx = y_copy == 0
    neg_idx = np.where(neg_idx)[0]

    pert_idx = np.random.choice(neg_idx, n_pert, replace=False)

    y_copy[pert_idx] = 1

    return y_copy


def make_gaussian_data(m0, m1, s0, s1, n, p0, p1, num_features=2, noise=0.0):
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


def make_trend_gaussian_data(m0, m1, s0, s1, n, num_features=2, noise=0.0, uniform_range=[-3, 3]):
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


def generate_gaussian_dataset(m0, m1, s0, s1, p0, p1):
    def wrapped(n_train, n_update, n_test, num_features=2, noise=0.0):
        x_train, y_train = make_gaussian_data(m0, m1, s0, s1, n_train, p0, p1, num_features=num_features, noise=noise)

        x_update, y_update = make_gaussian_data(m0, m1, s0, s1, n_update, p0, p1, num_features=num_features)
        x_test, y_test = make_gaussian_data(m0, m1, s0, s1, n_test, p0, p1, num_features=num_features)

        return x_train, y_train, x_update, y_update, x_test, y_test

    return wrapped


def generate_sklearn_make_classification_dataset(n_train, n_update, n_test, num_features=2, noise=0.0):
    x, y = make_classification(n_train + n_update + n_test, n_informative=num_features, n_features=num_features,
                               n_classes=2, n_clusters_per_class=2, n_redundant=0, flip_y=0, class_sep=1.0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_update + n_test)
    x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

    if noise > 0.0:
        neg_idx = y_train == 0
        bernoulli = np.random.choice([False, True], len(y_train), p=[1 - noise, noise])
        neg_idx = np.logical_and(neg_idx, bernoulli)
        neg_idx = np.where(neg_idx)
        y_train[neg_idx] = 1

    return x_train, y_train, x_update, y_update, x_test, y_test


def generate_moons_dataset(n_train, n_update, n_test, num_features=2, noise=0.0):
    x_train, y_train = make_moons(n_train, noise=noise)
    x_update, y_update = make_moons(n_update, noise=noise)
    x_test, y_test = make_moons(n_test, noise=noise)
    #     x, y = make_moons(n_train + n_update + n_test, noise=noise)
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_update + n_test)
    #     x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

    return x_train, y_train, x_update, y_update, x_test, y_test


def generate_circles_dataset(n_train, n_update, n_test, num_features=2, noise=0.0):
    x, y = make_moons(n_train + n_update + n_test, noise=noise)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_update + n_test)
    x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test)

    return x_train, y_train, x_update, y_update, x_test, y_test


def generate_gaussian_quantile_dataset(n_train, n_update, n_test, num_features=2, noise=0.0):
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


def load_mimiciii_data():
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
        'd_name': 'mimiciii',
    }

    return dataset


def generate_mimic_dataset():
    data = load_mimiciii_data()

    float_cols = []

    for i, column in enumerate(data["X"].columns):
        if "float" in str(data["X"][column].dtype):
            float_cols.append(i)

    float_cols = np.array(float_cols)

    x = data["X"].to_numpy()
    y = data["y"].to_numpy()

    def wrapped(n_train, n_update, n_test, **kwargs):
        n_train = int(len(y) * n_train)
        n_update = int(len(y) * n_update)
        n_test = int(len(y) * n_test)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_update + n_test, stratify=y)
        x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test, stratify=y_test)

        data_mean = np.mean(x_train[:, float_cols], 0)
        data_std = np.std(x_train[:, float_cols], 0)

        x_train[:, float_cols] = (x_train[:, float_cols] - data_mean) / data_std
        x_update[:, float_cols] = (x_update[:, float_cols] - data_mean) / data_std
        x_test[:, float_cols] = (x_test[:, float_cols] - data_mean) / data_std

        return x_train, y_train, x_update, y_update, x_test, y_test

    return wrapped


def load_support2cls_data():
    df = pd.read_csv('support2.csv')
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