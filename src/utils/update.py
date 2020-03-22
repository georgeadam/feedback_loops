import copy
import numpy as np

from src.utils.misc import create_empty_rates
from src.utils.metrics import compute_all_rates


def wrapped(fn, **kwargs):
    def inside(*args, **specified_args):
        return fn(*args, **kwargs, **specified_args)

    return inside


def get_update_fn(update_type, temporal=False):
    if temporal:
        update_fn = update_model_temporal
    else:
        update_fn = update_model_generic

    if update_type == "feedback_online_single_batch":
        return wrapped(update_fn, cumulative_data=False, include_train=False, weight_type=None,
                       full_fit=False, feedback=True)
    elif update_type == "feedback_online_all_update_data":
        return wrapped(update_fn, cumulative_data=True, include_train=False, weight_type=None,
                       full_fit=False, feedback=True)
    elif update_type == "feedback_online_all_data":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                       full_fit=False, feedback=True)
    elif update_type == "feedback_full_fit":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                full_fit=True, feedback=True)
    elif update_type == "feedback_online_all_update_data_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=False, weight_type=None,
                       full_fit=False, feedback=True)
    elif update_type == "feedback_online_all_data_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="linearly_decreasing",
                       full_fit=False, feedback=True)
    elif update_type == "feedback_full_fit_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="linearly_decreasing",
                full_fit=True, feedback=True)
    elif update_type == "no_feedback_online_single_batch":
        return wrapped(update_fn, cumulative_data=False, include_train=False, weight_type=None,
                       full_fit=False, feedback=False)
    elif update_type == "no_feedback_online_all_update_data":
        return wrapped(update_fn, cumulative_data=True, include_train=False, weight_type=None,
                       full_fit=False, feedback=False)
    elif update_type == "no_feedback_online_all_data":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                       full_fit=False, feedback=False)
    elif update_type == "no_feedback_full_fit":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                full_fit=True, feedback=False)
    elif update_type == "no_feedback_online_all_update_data_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=False, weight_type="linearly_decreasing",
                       full_fit=False, feedback=False)
    elif update_type == "no_feedback_online_all_data_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="linearly_decreasing",
                       full_fit=False, feedback=False)
    elif update_type == "no_feedback_full_fit_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="linearly_decreasing",
                full_fit=True, feedback=False)
    elif update_type == "feedback_full_fit_confidence":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="confidence",
                       full_fit=True, feedback=True)
    elif update_type == "feedback_full_fit_drop":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="drop",
                       full_fit=True, feedback=True)
    elif update_type == "feedback_full_fit_partial_confidence":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="partial_confidence",
                       full_fit=True, feedback=True)
    elif update_type == "no_feedback_full_fit_confidence":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="confidence",
                       full_fit=True, feedback=False)
    elif update_type == "no_feedback_full_fit_drop":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="drop",
                       full_fit=True, feedback=False)
    elif update_type == "no_feedback_full_fit_partial_confidence":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="partial_confidence",
                       full_fit=True, feedback=False)
    elif update_type == "evaluate":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                       full_fit=True, feedback=False, update=False)


def map_update_type(update_type):
    if update_type.startswith("feedback_online_single_batch"):
        return "ssrd"
    elif update_type.startswith("no_feedback_full_fit_confidence"):
        return "no_feedback_confidence"
    elif update_type.startswith("no_feedback_full_fit_drop"):
        return "no_feedback_drop"
    elif update_type.startswith("no_feedback_full_fit_partial_confidence"):
        return "no_feedback_partial_confidence"
    elif update_type.startswith("feedback_full_fit_confidence"):
        return "cad_confidence"
    elif update_type.startswith("feedback_full_fit_drop"):
        return "cad_drop"
    elif update_type.startswith("feedback_full_fit_partial_confidence"):
        return "cad_partial_confidence"
    elif update_type.startswith("feedback_online_all_update_data"):
        return "ssad-nt"
    elif update_type.startswith("feedback_online_all_data"):
        return "ssad-t"
    elif update_type.startswith("feedback_full_fit"):
        return "cad"
    elif update_type.startswith("no_feedback"):
        return "no_feedback"
    elif update_type.startswith("evaluate"):
        return "static"


def update_model_noise(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                       intermediate=False, threshold=None, rate=0.2, dynamic_desired_rate=None, dynamic_desired_value=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)

    rates = create_empty_rates()

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        neg_idx = np.where(sub_y == 0)[0]

        if len(neg_idx) > 0:
            fp_idx = np.random.choice(neg_idx, int(rate * len(sub_y)))
            sub_y[fp_idx] = 1

        new_model.partial_fit(sub_x, sub_y, classes)

        sub_prob = new_model.predict_proba(x_train)

        if dynamic_desired_rate is not None:
            threshold = find_threshold(y_train, sub_prob, dynamic_desired_rate, dynamic_desired_value)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_conditional_trust(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                   intermediate=False, threshold=None, physician_fpr=0.1,
                                   dynamic_desired_rate=None, dynamic_desired_value=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)

    trust = None

    trusts = []
    rates = create_empty_rates()

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1

        if trust is None:
            trust = 1 - float(len(model_fp_idx)) / float(len(sub_y))

        trusts.append(trust)

        physician_pred = copy.deepcopy(sub_y)
        neg_idx = np.where(physician_pred == 0)[0]
        physician_fp_idx = np.random.choice(neg_idx, int(physician_fpr * len(sub_y)))
        physician_pred[physician_fp_idx] = 1

        bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])

        target = bernoulli * model_pred + (1 - bernoulli) * physician_pred

        new_model.partial_fit(sub_x, target, classes)
        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]

        fpr = float(len(model_fp_idx)) / float(len(sub_y))

        sub_prob = new_model.predict_proba(x_train)

        if dynamic_desired_rate is not None:
            threshold = find_threshold(y_train, sub_prob, dynamic_desired_rate, dynamic_desired_value)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

        trust = 1 - fpr

    return new_model, rates, trusts


def update_model_increasing_trust(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                  intermediate=False, threshold=None, trusts=[], physician_fpr=0.1,
                                  dynamic_desired_rate=None, dynamic_desired_value=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)
    rates = create_empty_rates()

    for i in range(num_updates):
        trust = trusts[i]

        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1

        physician_pred = copy.deepcopy(sub_y)
        neg_idx = np.where(physician_pred == 0)[0]

        if len(neg_idx) > 0:
            physician_fp_idx = np.random.choice(neg_idx, min(len(neg_idx), int(physician_fpr * len(sub_y))))
            physician_pred[physician_fp_idx] = 1

        bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])

        target = bernoulli * model_pred + (1 - bernoulli) * physician_pred

        new_model.partial_fit(sub_x, target, classes)
        model_pred = new_model.predict(sub_x)

        sub_prob = new_model.predict_proba(x_train)

        if dynamic_desired_rate is not None:
            threshold = find_threshold(y_train, sub_prob, dynamic_desired_rate, dynamic_desired_value)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_generic(model, x_train, y_train, x_update, y_update, x_test, y_test,
                         num_updates, cumulative_data=False, include_train=False, weight_type=None, full_fit=False,
                         feedback=False, update=True, intermediate=False, threshold=None, dynamic_desired_rate=None,
                         dynamic_desired_value=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)
    cumulative_x = None
    cumulative_y = None

    rates = create_empty_rates()
    meta_weights = initialize_weights(weight_type, x_train, include_train)
    weights = initialize_weights(weight_type, x_train, include_train)

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])
        sub_conf = new_model.predict_proba(sub_x)[:, 1]
        replace_labels(feedback, new_model, sub_x, sub_y, threshold)

        cumulative_x, cumulative_y = build_cumulative_data(cumulative_data, cumulative_x, cumulative_y, sub_x, sub_y)
        weights = get_weights(weight_type, meta_weights, weights, x_train, cumulative_x, sub_conf, sub_y, threshold, include_train)
        make_update(cumulative_x, cumulative_y, full_fit, include_train, new_model, update, weights, x_train, y_train)

        sub_prob = new_model.predict_proba(x_train)

        if dynamic_desired_rate is not None:
            threshold = find_threshold(y_train, sub_prob, dynamic_desired_rate, dynamic_desired_value)

        loss = new_model.evaluate(x_test, y_test)
        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)
        rates["loss"].append(loss)

    return new_model, rates


def update_model_temporal(model, x_train, y_train, x_rest, y_rest, years, train_year_limit=1999, update_year_limit=2019,
                          cumulative_data=False, include_train=False, weight_type=None, full_fit=False,
                          feedback=False, update=True, next_year=True, intermediate=False, threshold=None, dynamic_desired_rate=None,
                          dynamic_desired_value=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    cumulative_x = None
    cumulative_y = None

    rates = create_empty_rates()

    meta_weights = initialize_weights(weight_type, x_train, include_train)
    weights = initialize_weights(weight_type, x_train, include_train)

    for year in range(train_year_limit + 1, update_year_limit):
        sub_idx = years == year

        if next_year:
            test_idx = years == year + 1
        else:
            test_idx = years == update_year_limit

        sub_x = x_rest[sub_idx]
        sub_y = copy.deepcopy(y_rest[sub_idx])
        sub_conf = new_model.predict_proba(sub_x)[:, 1]
        replace_labels(feedback, new_model, sub_x, sub_y, threshold)

        cumulative_x, cumulative_y = build_cumulative_data(cumulative_data, cumulative_x, cumulative_y, sub_x, sub_y)
        weights = get_weights(weight_type, meta_weights, weights, x_train, cumulative_x, sub_conf, sub_y, threshold, include_train)
        make_update(cumulative_x, cumulative_y, full_fit, include_train, new_model, update, weights, x_train, y_train)

        sub_prob = new_model.predict_proba(x_train)

        if dynamic_desired_rate is not None:
            threshold = find_threshold(y_train, sub_prob, dynamic_desired_rate, dynamic_desired_value)

        loss = new_model.evaluate(x_rest[test_idx], y_rest[test_idx])
        append_rates(intermediate, new_model, rates, threshold, x_rest[test_idx], y_rest[test_idx])
        rates["loss"].append(loss)

    return new_model, rates


def replace_labels(feedback, new_model, sub_x, sub_y, threshold, trust_fn=None, clinician_fpr=0.0):
    if feedback:
        sub_pred = new_model.predict_proba(sub_x)
        sub_pred = sub_pred[:, 1] > threshold

        fp_idx = np.logical_and(sub_y == 0, sub_pred == 1)
        sub_y[fp_idx] = 1


def make_update(cumulative_x, cumulative_y, full_fit, include_train, new_model, update, weights, x_train, y_train):
    if update:
        if include_train:
            if full_fit:
                new_model.fit(np.concatenate((cumulative_x, x_train)), np.concatenate((cumulative_y, y_train)),
                              weights)
            else:
                new_model.partial_fit(np.concatenate((cumulative_x, x_train)), np.concatenate((cumulative_y, y_train)),
                                      weights)
        else:
            if full_fit:
                new_model.fit(cumulative_x, cumulative_y, weights)
            else:
                new_model.partial_fit(cumulative_x, cumulative_y, weights)


def append_rates(intermediate, new_model, rates, threshold, x_test, y_test):
    if intermediate:
        if threshold is None:
            pred_prob = new_model.predict_proba(x_test)
            new_pred = new_model.predict(x_test)
        else:
            pred_prob = new_model.predict_proba(x_test)
            new_pred = pred_prob[:, 1] >= threshold

        updated_rates = compute_all_rates(y_test, new_pred, pred_prob)

        for key in rates.keys():
            if key in updated_rates.keys():
                rates[key].append(updated_rates[key])


def find_threshold(y, y_prob, desired_rate, desired_value, tol=0.01):
    thresholds = np.linspace(0.01, 0.999, 1000)
    best_threshold = None
    best_diff = float("inf")

    l = 0
    r = len(thresholds)

    while l < r:
        mid = l + (r - l) // 2
        threshold = thresholds[mid]

        temp_pred = y_prob[:, 1] >= threshold
        rates = compute_all_rates(y, temp_pred, y_prob)

        direction = get_direction(desired_rate)

        if abs(rates[desired_rate] - desired_value) <= desired_value * tol:
            return threshold
        elif (rates[desired_rate] < desired_value and direction == "decreasing") or (rates[desired_rate] > desired_value and direction == "increasing"):
            l = mid + 1
        else:
            r = mid - 1

        if abs(rates[desired_rate] - desired_value) <= best_diff:
            best_diff = abs(rates[desired_rate] - desired_value)
            best_threshold = threshold


    return best_threshold


def get_direction(rate):
    if rate == "fpr" or rate == "tnr":
        return "increasing"
    elif rate == "fnr" or rate == "tpr":
        return "decreasing"


def initialize_weights(weight_type, x_train, include_train):
    if weight_type is None:
        weights = None
    elif include_train:
        weights = np.array(np.ones(len(x_train)))
    else:
        weights = np.array([]).astype(float)

    return weights


def get_weights(weight_type, meta_weights, prev_weights, x_train, cumulative_x, sub_conf, sub_y, threshold, include_train):
    if weight_type == "linearly_decreasing":
        if include_train:
            weights = np.concatenate((np.ones(len(cumulative_x)), np.ones(len(x_train))))
        else:
            weights = np.ones(len(cumulative_x))

        meta_weights = np.concatenate((np.zeros(len(sub_conf)), meta_weights))
        meta_weights += 1
        weights = weights / meta_weights
    elif weight_type == "confidence":
        sub_idx = sub_y == 0
        sub_weights = copy.deepcopy(sub_conf)
        sub_weights[sub_idx] = 1
        weights = np.concatenate([sub_weights, prev_weights])
    elif weight_type == "drop":
        sorted_idx = np.argsort(sub_conf)
        unsorted_idx = np.argsort(sorted_idx)
        sorted_y = sub_y[sorted_idx]
        sorted_pos_idx = np.where(sorted_y == 1)[0]
        sorted_pos_idx = sorted_pos_idx[: int(0.1 * len(sorted_pos_idx))]

        temp_idx = unsorted_idx[sorted_pos_idx]
        pos_idx = np.where(sub_conf[temp_idx] > threshold)[0]
        pos_idx = temp_idx[pos_idx]
        neg_idx = np.delete(np.arange(len(sub_y)), pos_idx)
        sub_weights = copy.deepcopy(sub_conf)
        sub_weights[neg_idx] = 1
        sub_weights[pos_idx] = 0

        weights = np.concatenate([sub_weights, prev_weights])
    elif weight_type == "partial_confidence":
        sorted_idx = np.argsort(sub_conf)
        unsorted_idx = np.argsort(sorted_idx)
        sorted_y = sub_y[sorted_idx]
        sorted_pos_idx = np.where(sorted_y == 1)[0]
        sorted_pos_idx = sorted_pos_idx[: int(0.1 * len(sorted_pos_idx))]

        temp_idx = unsorted_idx[sorted_pos_idx]
        pos_idx = np.where(sub_conf[temp_idx] > threshold)[0]
        pos_idx = temp_idx[pos_idx]
        neg_idx = np.delete(np.arange(len(sub_y)), pos_idx)
        sub_weights = copy.deepcopy(sub_conf)
        sub_weights[neg_idx] = 1
        sub_weights[pos_idx] = sub_conf[pos_idx]

        weights = np.concatenate([sub_weights, prev_weights])
    else:
        weights = None

    return weights


def build_cumulative_data(cumulative_data, cumulative_x, cumulative_y, sub_x, sub_y):
    if cumulative_x is None or not cumulative_data:
        cumulative_x = sub_x
        cumulative_y = sub_y
    else:
        cumulative_x = np.concatenate((sub_x, cumulative_x))
        cumulative_y = np.concatenate((sub_y, cumulative_y))

    return cumulative_x, cumulative_y