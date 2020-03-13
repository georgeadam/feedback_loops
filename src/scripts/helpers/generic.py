import numpy as np

from src.utils.metrics import compute_all_rates
from src.utils.misc import create_empty_rates
from src.utils.rand import set_seed
from src.utils.update import find_threshold


def train_update_loop(model_fn, n_train, n_update, n_test, num_updates, num_features,
                      initial_desired_rate, initial_desired_value, dynamic_desired_rate, data_fn, update_fn, bad_model, worst_case,
                      seeds):
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

        if not bad_model:
            model.fit(x_train, y_train)
            loss = model.evaluate(x_test, y_test)

        y_prob = model.predict_proba(x_train)

        if initial_desired_rate is not None:
            threshold = find_threshold(y_train, y_prob, initial_desired_rate, initial_desired_value)
        else:
            threshold = 0.5

        if worst_case:
            update_y_prob = model.predict_proba(x_update)
            update_y_pred = update_y_prob[:, 1] > threshold

            update_fps = np.logical_and(update_y_pred == 1, y_update == 0).astype(int)
            sorted_idx = np.argsort(-update_fps)
            x_update = x_update[sorted_idx]
            y_update = y_update[sorted_idx]

        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold

        initial_rates = compute_all_rates(y_test, y_pred, y_prob)
        initial_rates["loss"] = loss

        y_prob = model.predict_proba(x_train)
        y_pred = y_prob[:, 1] > threshold
        temp_train_rates = compute_all_rates(y_train, y_pred, y_prob)
        dynamic_desired_value = get_dyanmic_desired_value(dynamic_desired_rate, temp_train_rates)

        new_model, updated_rates = update_fn(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                          intermediate=True, threshold=threshold,
                                          dynamic_desired_rate=dynamic_desired_rate,
                                          dynamic_desired_value=dynamic_desired_value)

        for key in rates.keys():
            rates[key].append([initial_rates[key]] + updated_rates[key])
            stats["initial"][key].append(initial_rates[key])
            stats["updated"][key].append(updated_rates[key][-1])

    return rates, stats


def gold_standard_loop(model_fn, n_train, n_update, n_test, num_features, desired_rate, desired_value, data_fn, seeds):
    seeds = np.arange(seeds)
    rates = create_empty_rates()

    for seed in seeds:
        np.random.seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)

        model = model_fn(num_features=x_train.shape[1])
        model.fit(np.concatenate((x_train, x_update)), np.concatenate((y_train, y_update)))
        loss = model.evaluate(x_test, y_test)
        y_prob = model.predict_proba(np.concatenate((x_train, x_update)))

        if desired_rate is not None:
            threshold = find_threshold(np.concatenate((y_train, y_update)), y_prob, desired_rate, desired_value)
        else:
            threshold = 0.5

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