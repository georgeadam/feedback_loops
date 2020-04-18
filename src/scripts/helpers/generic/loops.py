import copy
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.data import TEMPORAL_DATA_TYPES
from src.utils.metrics import compute_all_rates
from src.utils.misc import create_empty_rates
from src.utils.preprocess import get_scaler
from src.utils.rand import set_seed
from src.utils.threshold import find_threshold

from typing import Dict, Callable, SupportsFloat


def train_update_loop_static(data_fn: Callable=None, model_fn: Callable=None, update_fn: Callable=None,
                             n_train: SupportsFloat=0.1, n_update: SupportsFloat=0.7, n_test: SupportsFloat=0.2,
                             num_features: int=20, num_updates: int=100, idr: str= "fpr", idv: float=0.1, tvp: float=0.0,
                             ddr: str=None, ddp: str= "train", worst_case: bool=False, seeds: int=1,
                             clinician_fpr: float=0.0, clinician_trust: float=1.0, normalize: bool=True,
                             **kwargs) -> (Dict, Dict):
    seeds = np.arange(seeds)
    rates = create_empty_rates()

    stats = {"updated": {key: [] for key in rates.keys()},
             "initial": {key: [] for key in rates.keys()}}

    for seed in seeds:
        print(seed)
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test, cols = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)
        scaler = get_scaler(normalize, cols)
        scaler.fit(x_train)
        model = model_fn(num_features=x_train.shape[1])

        if tvp > 0:
            x_train_fit, x_threshold_fit, y_train_fit, y_threshold_fit = train_test_split(x_train, y_train, stratify=y_train,
                                                                                          test_size=tvp)
        else:
            x_train_fit, x_threshold_fit, y_train_fit, y_threshold_fit = x_train, x_train, y_train, y_train

        set_seed(seed)
        model.fit(scaler.transform(x_train_fit), y_train_fit)
        loss = model.evaluate(scaler.transform(x_test), y_test)

        y_prob = model.predict_proba(scaler.transform(x_threshold_fit))

        if idr is not None:
            threshold = find_threshold(y_threshold_fit, y_prob, idr, idv)
        else:
            threshold = 0.5

        if worst_case:
            update_y_prob = model.predict_proba(scaler.transform(x_update))
            update_y_pred = update_y_prob[:, 1] > threshold

            update_fps = np.logical_and(update_y_pred == 1, y_update == 0).astype(int)
            sorted_idx = np.argsort(-update_fps)
            x_update = x_update[sorted_idx]
            y_update = y_update[sorted_idx]

        set_seed(seed)
        model.fit(scaler.transform(x_train), y_train)
        y_prob = model.predict_proba(scaler.transform(x_test))
        y_pred = y_prob[:, 1] > threshold

        initial_rates = compute_all_rates(y_test, y_pred, y_prob, initial=True)
        initial_rates["loss"] = loss

        ddv = get_dyanmic_desired_value(ddr, initial_rates)

        new_model, updated_rates = update_fn(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                             intermediate=True, threshold=threshold, ddr=ddr, ddv=ddv, ddp=ddp,
                                             clinician_fpr=clinician_fpr, clinician_trust=clinician_trust,
                                             scaler=scaler)

        for key in rates.keys():
            rates[key].append([initial_rates[key]] + updated_rates[key])
            stats["initial"][key].append(initial_rates[key])
            stats["updated"][key].append(updated_rates[key][-1])

    return rates, stats


def gold_standard_loop(model_fn, n_train, n_update, n_test, num_features, desired_rate, desired_value,
                       threshold_validation_percentage, data_fn, seeds):
    seeds = np.arange(seeds)
    rates = create_empty_rates()

    for seed in seeds:
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)

        model = model_fn(num_features=x_train.shape[1])
        concat_x = np.concatenate((x_train, x_update))
        concat_y = np.concatenate((y_train, y_update))

        if threshold_validation_percentage > 0:
            concat_x_train_fit, concat_x_threshold_fit, concat_y_train_fit, concat_y_threshold_fit = train_test_split(concat_x, concat_y, stratify=concat_y,
                                                                                          test_size=threshold_validation_percentage)
        else:
            concat_x_train_fit, concat_x_threshold_fit, concat_y_train_fit, concat_y_threshold_fit = concat_x, concat_x, concat_y, concat_y

        set_seed(seed)
        model.fit(concat_x_train_fit, concat_y_train_fit)
        loss = model.evaluate(x_test, y_test)
        y_prob = model.predict_proba(concat_x_threshold_fit)

        if desired_rate is not None:
            threshold = find_threshold(concat_y_threshold_fit, y_prob, desired_rate, desired_value)
        else:
            threshold = 0.5

        set_seed(seed)
        model.fit(concat_x, concat_y)
        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold
        gold_standard_rates = compute_all_rates(y_test, y_pred, y_prob)
        gold_standard_rates["loss"] = loss

        for key in rates.keys():
            rates[key].append(gold_standard_rates[key])

    return rates


def get_dyanmic_desired_value(desired_dynamic_rate, rates):
    if desired_dynamic_rate is not None:
        return rates[desired_dynamic_rate]

    return None


def train_update_loop_temporal(data_fn: Callable=None, model_fn: Callable=None, update_fn: Callable=None,
                               tyl: int=1999, uyl: int=2019, idr: str= "fpr", idv: float=0.1, tvp: float=0.0,
                               ddr: str=None, ddp: str= "train", next_year: bool=True,
                               seeds: int=1, clinician_fpr: float=0.0, clinician_trust: float=1.0,
                               normalize: bool=True, **kwargs):
    seeds = np.arange(seeds)
    rates = create_empty_rates()

    stats = {"updated": {key: [] for key in rates.keys()},
             "initial": {key: [] for key in rates.keys()}}

    for seed in seeds:
        print(seed)
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test, cols = data_fn(0.3, 0.4, 0.3, num_features=0)
        x = np.concatenate([x_train, x_update, x_test], axis=0)
        y = np.concatenate([y_train, y_update, y_test])


        model = model_fn(num_features=x.shape[1] - 1)

        train_idx = x[:, 0] <= tyl
        x_train, y_train = x[train_idx], y[train_idx]
        x_rest, y_rest = x[~train_idx], y[~train_idx]
        years_rest = x_rest[:, 0]
        x_train, x_rest = x_train[:, 1:], x_rest[:, 1:]

        scaler = get_scaler(normalize, cols)

        scaler.fit(x_train)

        if next_year:
            eval_idx = years_rest == tyl + 1
        else:
            eval_idx = years_rest == uyl

        x_eval, y_eval = x_rest[eval_idx],  y_rest[eval_idx]

        if tvp > 0:
            x_train_fit, x_threshold_fit, y_train_fit, y_threshold_fit = train_test_split(x_train, y_train, stratify=y_train,
                                                                                          test_size=tvp)
        else:
            x_train_fit, x_threshold_fit, y_train_fit, y_threshold_fit = x_train, x_train, y_train, y_train

        set_seed(seed)
        model.fit(scaler.transform(x_train_fit), y_train_fit)
        loss = model.evaluate(scaler.transform(x_eval), y_eval)

        y_prob = model.predict_proba(scaler.transform(x_threshold_fit))

        if idr is not None:
            threshold = find_threshold(y_threshold_fit, y_prob, idr, idv)
        else:
            threshold = 0.5

        set_seed(seed)
        model.fit(scaler.transform(x_train), y_train)
        y_prob = model.predict_proba(scaler.transform(x_eval))
        y_pred = y_prob[:, 1] > threshold

        initial_rates = compute_all_rates(y_eval, y_pred, y_prob, initial=True)
        initial_rates["loss"] = loss

        ddv = get_dyanmic_desired_value(ddr, initial_rates)

        new_model, updated_rates = update_fn(model, x_train, y_train, x_rest, y_rest, years_rest, tyl, uyl,
                                             next_year=next_year, intermediate=True, threshold=threshold, ddr=ddr,
                                             ddv=ddv, ddp=ddp, clinician_fpr=clinician_fpr,
                                             clinician_trust=clinician_trust, scaler=scaler)

        for key in rates.keys():
            rates[key].append([initial_rates[key]] + updated_rates[key])
            stats["initial"][key].append(initial_rates[key])
            stats["updated"][key].append(updated_rates[key][-1])

    return rates, stats


def call_update_loop(args, data_fn, model_fn, update_fn):
    if args.data.temporal:
        return train_update_loop_temporal(data_fn, model_fn, update_fn, tyl=args.data.tyl, uyl=args.data.uyl,
                                          idr=args.rates.idr, idv=args.rates.idv, tvp=args.rates.tvp, ddr=args.rates.ddr,
                                          ddp=args.rates.ddp, next_year=args.data.next_year, seeds=args.misc.seeds,
                                          clinician_fpr=args.rates.clinician_fpr, clinician_trust=args.rates.clinician_trust,
                                          normalize=args.data.normalize)
    else:
        return train_update_loop_static(data_fn, model_fn, update_fn, n_train=args.data.n_train, n_update=args.data.n_update,
                                        n_test=args.data.n_test, num_features=args.data.num_features, num_updates=args.data.num_updates,
                                        idr=args.rates.idr, idv=args.rates.idv, tvp=args.rates.tvp, ddr=args.rates.ddr,
                                        ddp=args.rates.ddp, worst_case=args.data.worst_case, seeds=args.misc.seeds,
                                        clinician_fpr=args.rates.clinician_fpr, clinician_trust=args.rates.clinician_trust,
                                        normalize=args.data.normalize)


