import copy
import numpy as np
import os
import pandas as pd
from scipy.stats import multivariate_normal

from sklearn.datasets import make_classification, make_gaussian_quantiles, make_moons
from sklearn.model_selection import train_test_split

from settings import ROOT_DIR


STATIC_DATA_TYPES = ["gaussian", "sklearn", "moons", "mimic_iii", "support2"]
TEMPORAL_DATA_TYPES = ["mimic_iv", "mimic_iv_12h", "mimic_iv_24h"]

mimic_iv_paths = {"mimic_iv": "mimic_iv_datasets_with_year_imputed.csv",
                  "mimic_iv_12h": "mimic_iv_datasets_with_year_12hrs_imputed.csv",
                  "mimic_iv_24h": "mimic_iv_datasets_with_year_24hrs_imputed.csv"}

def get_data_fn(args):
    if args.data_type == "gaussian":
        if hasattr(args, "m0"):
            return generate_gaussian_dataset(args.m0, args.m1, args.s0, args.s1, args.p0, args.p1)
        else:
            return generate_gaussian_dataset()
    elif args.data_type == "sklearn":
        return generate_sklearn_make_classification_dataset
    elif args.data_type == "mimic_iii":
        return generate_real_dataset(load_mimic_iii_data, args.sorted, balanced=args.balanced)
    elif "mimic_iv" in args.data_type:
        return generate_real_dataset(load_mimic_iv_data, args.sorted,
                                     mimic_iv_paths[args.data_type], args.balanced, temporal=args.temporal)
    elif args.data_type == "moons":
        return generate_moons_dataset
    elif args.data_type == "support2":
        return generate_real_dataset(load_support2cls_data, args.sorted)


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


def generate_gaussian_dataset(m0=-1, m1=1, s0=1, s1=1, p0=0.5, p1=0.5):
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


def load_mimic_iii_data(*args, **kargs):
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


def generate_real_dataset(fn, sorted=False, path=None, balanced=False, temporal=True):
    data = fn(path)

    year_idx = None

    for i, column in enumerate(data["X"].columns):
        if column == "year":
            year_idx = i

    if "year" in data["X"].columns and temporal:
        years = np.unique(data["X"]["year"])
    else:
        years = None

    x = data["X"].to_numpy()
    y = data["y"].to_numpy()

    if sorted:
        sort_idx = np.argsort(x["year"])
        x = x[sort_idx]
        y = y[sort_idx]

    nan_idx = np.where(np.isnan(x))[0]
    x = np.delete(x, nan_idx, 0)
    y = np.delete(y, nan_idx, 0)

    print("a")

    def wrapped(n_train, n_update, n_test, **kwargs):
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

        n_train = int(len(y_copy) * n_train)
        n_update = int(len(y_copy) * n_update)
        n_test = int(len(y_copy) * n_test)

        if year_idx is not None and not temporal:
            np.delete(x_copy, year_idx, 1)

        x_train, x_test, y_train, y_test = train_test_split(x_copy, y_copy, test_size=n_update + n_test,
                                                            stratify=y_copy)
        x_update, x_test, y_update, y_test = train_test_split(x_test, y_test, test_size=n_test, stratify=y_test)

        return x_train, y_train, x_update, y_update, x_test, y_test

    return wrapped


def load_support2cls_data():
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


def load_mimic_iv_data(path):
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

    label = 'mort_icu'
    X_df = df_adult[train_cols]
    y_df = df_adult[label]

    dataset = {
        'problem': 'classification',
        'X': X_df,
        'y': y_df,
        'd_name': 'mimic_iv',
    }

    return dataset