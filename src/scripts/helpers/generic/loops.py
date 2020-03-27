import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.data import TEMPORAL_DATA_TYPES
from src.utils.metrics import compute_all_rates
from src.utils.misc import create_empty_rates
from src.utils.rand import set_seed
from src.utils.update import find_threshold


def train_update_loop_static(model_fn=None, n_train=0.1, n_update=0.7, n_test=0.2, num_updates=100, num_features=20,
                             initial_desired_rate="fpr", initial_desired_value=0.1, threshold_validation_percentage=0.0,
                             clinician_fpr=0.0, dynamic_desired_rate=None, dynamic_desired_partition="train",
                             data_fn=None, update_fn=None, bad_model=False, worst_case=False,
                             seeds=1, clinician_trust=1.0, **kwargs):
    seeds = np.arange(seeds)

    rates = create_empty_rates()

    stats = {"updated": {key: [] for key in rates.keys()},
             "initial": {key: [] for key in rates.keys()}}

    for seed in seeds:
        print(seed)
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)

        model = model_fn(num_features=x_train.shape[1])

        if threshold_validation_percentage > 0:
            x_train_fit, x_threshold_fit, y_train_fit, y_threshold_fit = train_test_split(x_train, y_train, stratify=y_train,
                                                                                          test_size=threshold_validation_percentage)
        else:
            x_train_fit, x_threshold_fit, y_train_fit, y_threshold_fit = x_train, x_train, y_train, y_train

        if not bad_model:
            set_seed(seed)
            model.fit(x_train_fit, y_train_fit)
            loss = model.evaluate(x_test, y_test)

        y_prob = model.predict_proba(x_threshold_fit)

        if initial_desired_rate is not None:
            threshold = find_threshold(y_threshold_fit, y_prob, initial_desired_rate, initial_desired_value)
        else:
            threshold = 0.5

        if worst_case:
            update_y_prob = model.predict_proba(x_update)
            update_y_pred = update_y_prob[:, 1] > threshold

            update_fps = np.logical_and(update_y_pred == 1, y_update == 0).astype(int)
            sorted_idx = np.argsort(-update_fps)
            x_update = x_update[sorted_idx]
            y_update = y_update[sorted_idx]

        set_seed(seed)
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold

        initial_rates = compute_all_rates(y_test, y_pred, y_prob)
        initial_rates["loss"] = loss

        dynamic_desired_value = get_dyanmic_desired_value(dynamic_desired_rate, initial_rates)

        new_model, updated_rates = update_fn(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                             intermediate=True, threshold=threshold,
                                             dynamic_desired_rate=dynamic_desired_rate,
                                             dynamic_desired_value=dynamic_desired_value,
                                             dynamic_desired_partition=dynamic_desired_partition,
                                             clinician_fpr=clinician_fpr,
                                             clinician_trust=clinician_trust)

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


def train_update_loop_temporal(model_fn=None, train_year_limit=1999, update_year_limit=2019, initial_desired_rate="fpr",
                               initial_desired_value=0.1, threshold_validation_percentage=0.0, clinician_fpr=0.0, dynamic_desired_rate=None,
                               dynamic_desired_partition="train", data_fn=None, update_fn=None, bad_model=False,
                               next_year=True, seeds=1, clinician_trust=1.0, **kwargs):
    seeds = np.arange(seeds)

    rates = create_empty_rates()

    stats = {"updated": {key: [] for key in rates.keys()},
             "initial": {key: [] for key in rates.keys()}}

    for seed in seeds:
        print(seed)
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(0.3, 0.4, 0.3, num_features=0)
        x = np.concatenate([x_train, x_update, x_test], axis=0)
        y = np.concatenate([y_train, y_update, y_test])

        model = model_fn(num_features=x.shape[1] - 1)
        train_idx = x[:, 0] <= train_year_limit
        x_train, y_train = x[train_idx], y[train_idx]
        x_rest, y_rest = x[~train_idx], y[~train_idx]

        if next_year:
            eval_idx = x[:, 0] == train_year_limit + 1
        else:
            eval_idx = x[:, 0] == update_year_limit

        x_eval, y_eval = x[eval_idx],  y[eval_idx]

        if threshold_validation_percentage > 0:
            x_train_fit, x_threshold_fit, y_train_fit, y_threshold_fit = train_test_split(x_train, y_train, stratify=y_train,
                                                                                          test_size=threshold_validation_percentage)
        else:
            x_train_fit, x_threshold_fit, y_train_fit, y_threshold_fit = x_train, x_train, y_train, y_train

        if not bad_model:
            set_seed(seed)
            model.fit(x_train_fit[:, 1:], y_train_fit)
            loss = model.evaluate(x_eval[:, 1:], y_eval)

        y_prob = model.predict_proba(x_threshold_fit[:, 1:])

        if initial_desired_rate is not None:
            threshold = find_threshold(y_threshold_fit, y_prob, initial_desired_rate, initial_desired_value)
        else:
            threshold = 0.5

        set_seed(seed)
        model.fit(x_train[:, 1:], y_train)
        y_prob = model.predict_proba(x_eval[:, 1:])
        y_pred = y_prob[:, 1] > threshold

        initial_rates = compute_all_rates(y_eval, y_pred, y_prob)
        initial_rates["loss"] = loss

        dynamic_desired_value = get_dyanmic_desired_value(dynamic_desired_rate, initial_rates)

        years = x_rest[:, 0]
        x_train = np.delete(x_train, 0, 1)
        x_rest = np.delete(x_rest, 0, 1)

        new_model, updated_rates = update_fn(model, x_train, y_train, x_rest, y_rest, years, train_year_limit, update_year_limit,
                                             next_year=next_year,
                                             intermediate=True, threshold=threshold,
                                             dynamic_desired_rate=dynamic_desired_rate,
                                             dynamic_desired_value=dynamic_desired_value,
                                             dynamic_desired_partition=dynamic_desired_partition,
                                             clinician_fpr=clinician_fpr,
                                             clinician_trust=clinician_trust)

        for key in rates.keys():
            rates[key].append([initial_rates[key]] + updated_rates[key])
            stats["initial"][key].append(initial_rates[key])
            stats["updated"][key].append(updated_rates[key][-1])

    return rates, stats


def get_update_loop(temporal):
    if temporal:
        return train_update_loop_temporal
    else:
        return train_update_loop_static


