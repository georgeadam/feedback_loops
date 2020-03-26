import copy
import numpy as np

from src.utils.misc import create_empty_rates
from src.utils.metrics import compute_all_rates
from src.utils.rand import set_seed
from src.utils.trust import full_trust, conditional_trust

from sklearn.model_selection import train_test_split


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
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_online_all_update_data":
        return wrapped(update_fn, cumulative_data=True, include_train=False, weight_type=None,
                       fit_type="partial_fil", feedback=True)
    elif update_type == "feedback_online_all_data":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_full_fit":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                fit_type="fit", feedback=True)
    elif update_type == "feedback_online_all_update_data_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=False, weight_type=None,
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_online_all_data_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="linearly_decreasing",
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_full_fit_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="linearly_decreasing",
                fit_type="fit", feedback=True)
    elif update_type == "no_feedback_online_single_batch":
        return wrapped(update_fn, cumulative_data=False, include_train=False, weight_type=None,
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_online_all_update_data":
        return wrapped(update_fn, cumulative_data=True, include_train=False, weight_type=None,
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_online_all_data":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_full_fit":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                fit_type="fit", feedback=False)
    elif update_type == "no_feedback_online_all_update_data_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=False, weight_type="linearly_decreasing",
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_online_all_data_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="linearly_decreasing",
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_full_fit_weighted":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="linearly_decreasing",
                fit_type="fit", feedback=False)
    elif update_type == "feedback_full_fit_confidence":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="confidence",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_drop":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="drop",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_partial_confidence":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="partial_confidence",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_oracle":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="oracle",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_conditional_trust":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                fit_type="fit", feedback=True, trust_fn=conditional_trust)
    elif update_type == "no_feedback_full_fit_confidence":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="confidence",
                       fit_type="fit", feedback=False)
    elif update_type == "no_feedback_full_fit_drop":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="drop",
                       fit_type="fit", feedback=False)
    elif update_type == "no_feedback_full_fit_partial_confidence":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type="partial_confidence",
                       fit_type="fit", feedback=False)
    elif update_type == "no_feedback_full_fit_oracle":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                       fit_type="fit", feedback=False)
    elif update_type == "evaluate":
        return wrapped(update_fn, cumulative_data=True, include_train=True, weight_type=None,
                       fit_type="fit", feedback=False, update=False)


def map_update_type(update_type):
    if update_type.startswith("feedback_online_single_batch"):
        return "ssrd"
    elif update_type.startswith("no_feedback_full_fit_confidence"):
        return "no_feedback_confidence"
    elif update_type.startswith("no_feedback_full_fit_drop"):
        return "no_feedback_drop"
    elif update_type.startswith("no_feedback_full_fit_partial_confidence"):
        return "no_feedback_partial_confidence"
    elif update_type.startswith("no_feedback_full_fit_oracle"):
        return "no_feedback_oracle"
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
    elif update_type.startswith("feedback_full_fit_conditional_trust"):
        return "cad_conditional_trust"
    elif update_type.startswith("feedback_full_fit_oracle"):
        return "cad_oracle"
    elif update_type.startswith("feedback_full_fit"):
        return "feedback"
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


def update_model_generic(model, x_train, y_train, x_update, y_update, x_test, y_test,
                         num_updates, cumulative_data=False, include_train=False, weight_type=None, fit_type="fit",
                         feedback=False, update=True, intermediate=False, trust_fn=full_trust, clinician_fpr=0.0, threshold=None, dynamic_desired_rate=None,
                         dynamic_desired_value=None, dynamic_desired_partition=None, threshold_validation_percentage=0.2):
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)
    cumulative_x_update = np.array([]).astype(float).reshape(0, x_train.shape[1])
    cumulative_y_update = np.array([]).astype(int)

    rates = create_empty_rates()
    train_weights = initialize_weights(weight_type, x_train, include_train)
    cumulative_update_weights = np.array([]).astype(float)
    initial_fpr = compute_initial_fpr(model, threshold, x_train, y_train)

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])
        sub_y_unmodified = copy.deepcopy(y_update[idx_start: idx_end])
        sub_conf = new_model.predict_proba(sub_x)[:, 1]

        if i == 0:
            model_fpr = initial_fpr
        else:
            model_fpr = rates["fpr"][-1]

        sub_y = replace_labels(feedback, new_model, sub_x, sub_y, threshold, trust_fn, clinician_fpr, model_fpr)
        sub_weights = get_weights(weight_type, sub_conf, sub_y, sub_y_unmodified, threshold)
        threshold = make_update(x_train, y_train, cumulative_x_update, cumulative_y_update, sub_x, sub_y, train_weights,
                                cumulative_update_weights, sub_weights, new_model, threshold, fit_type, update, include_train,
                                dynamic_desired_rate, dynamic_desired_value, dynamic_desired_partition, threshold_validation_percentage)

        loss = new_model.evaluate(x_test, y_test)
        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)
        rates["loss"][-1] = loss
        cumulative_x_update, cumulative_y_update = build_cumulative_data(cumulative_data, cumulative_x_update,
                                                                         cumulative_y_update, sub_x, sub_y)
        cumulative_update_weights = build_cumulative_weights(cumulative_data, cumulative_update_weights, sub_weights)

    return new_model, rates


def update_model_temporal(model, x_train, y_train, x_rest, y_rest, years, train_year_limit=1999, update_year_limit=2019,
                          cumulative_data=False, include_train=False, weight_type=None, fit_type="fit",
                          feedback=False, update=True, next_year=True, trust_fn=full_trust, clinician_fpr=0.0, intermediate=False, threshold=None, dynamic_desired_rate=None,
                          dynamic_desired_value=None, dynamic_desired_partition=None, threshold_validation_percentage=0.2):
    new_model = copy.deepcopy(model)

    cumulative_x_update = np.array([]).astype(float).reshape(0, x_train.shape[1])
    cumulative_y_update = np.array([]).astype(int)

    rates = create_empty_rates()

    train_weights = initialize_weights(weight_type, x_train, include_train)
    cumulative_update_weights = np.array([]).astype(float)
    initial_fpr = compute_initial_fpr(model, threshold, x_train, y_train)

    for year in range(train_year_limit + 1, update_year_limit):
        sub_idx = years == year

        if next_year:
            test_idx = years == year + 1
        else:
            test_idx = years == update_year_limit

        sub_x = x_rest[sub_idx]
        sub_y = copy.deepcopy(y_rest[sub_idx])
        sub_y_unmodified = copy.deepcopy(y_rest[sub_idx])
        sub_conf = new_model.predict_proba(sub_x)[:, 1]

        if year == train_year_limit + 1:
            model_fpr = initial_fpr
        else:
            model_fpr = rates["fpr"][-1]

        sub_y = replace_labels(feedback, new_model, sub_x, sub_y, threshold, trust_fn, clinician_fpr, model_fpr)
        sub_weights = get_weights(weight_type, sub_conf, sub_y, sub_y_unmodified, threshold)
        threshold = make_update(x_train, y_train, cumulative_x_update, cumulative_y_update, sub_x, sub_y, train_weights,
                                cumulative_update_weights, sub_weights, new_model, threshold, fit_type, update, include_train,
                                dynamic_desired_rate, dynamic_desired_value, dynamic_desired_partition, threshold_validation_percentage)

        loss = new_model.evaluate(x_rest[test_idx], y_rest[test_idx])
        append_rates(intermediate, new_model, rates, threshold, x_rest[test_idx], y_rest[test_idx])
        rates["loss"][-1] = (loss)
        cumulative_x_update, cumulative_y_update = build_cumulative_data(cumulative_data, cumulative_x_update,
                                                                         cumulative_y_update, sub_x, sub_y)
        cumulative_update_weights = build_cumulative_weights(cumulative_data, cumulative_update_weights, sub_weights)

    return new_model, rates


def compute_initial_fpr(model, threshold, x_train, y_train):
    temp_prob = model.predict_proba(x_train)[:, 1]
    temp_pred = temp_prob > threshold
    initial_fps = np.logical_and(temp_pred == 1, y_train == 0)
    initial_fpr = len(y_train[initial_fps]) / len(y_train)
    return initial_fpr


def make_update(x_train, y_train, cumulative_x_update, cumulative_y_update, sub_x, sub_y, train_weights,
                cumulative_update_weights, sub_weights, new_model, threshold, fit_type, update, include_train,
                dynamic_desired_rate, dynamic_desired_value, dynamic_desired_partition, threshold_validation_percentage):
    if update:
        if dynamic_desired_rate is not None:
            all_train_x, all_train_y, all_train_weights, all_valid_x, all_valid_y = split_validation_data(x_train, y_train, cumulative_x_update,
                                                                                                          cumulative_y_update, sub_x, sub_y, train_weights,
                                                                                                          cumulative_update_weights, sub_weights,
                                                                                                          dynamic_desired_partition, threshold_validation_percentage,
                                                                                                          include_train)

            getattr(new_model, fit_type)(all_train_x, all_train_y, all_train_weights)
            valid_prob = new_model.predict_proba(all_valid_x)

            threshold = find_threshold(all_valid_y, valid_prob, dynamic_desired_rate, dynamic_desired_value)

        all_x, all_y, all_weights = combine_data(x_train, y_train, cumulative_x_update,
                                                 cumulative_y_update, sub_x, sub_y, train_weights,
                                                 cumulative_update_weights, sub_weights, include_train)
        getattr(new_model, fit_type)(all_x, all_y, all_weights)

    return threshold

def replace_labels(feedback, new_model, sub_x, sub_y, threshold, trust_fn=None, clinician_fpr=0.0, model_fpr=0.2):
    if feedback:
        model_pred = new_model.predict_proba(sub_x)
        model_pred = model_pred[:, 1] > threshold
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1
    else:
        model_pred = copy.deepcopy(sub_y)


    trust = trust_fn(model_fpr)

    clinician_pred = copy.deepcopy(sub_y)
    neg_idx = np.where(clinician_pred == 0)[0]
    physician_fp_idx = np.random.choice(neg_idx, min(int(clinician_fpr * len(sub_y)), len(neg_idx)))
    clinician_pred[physician_fp_idx] = 1

    bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])

    target = bernoulli * model_pred + (1 - bernoulli) * clinician_pred

    return target


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
        weights = np.array(np.ones(len(x_train)))
    elif include_train:
        weights = np.array(np.ones(len(x_train)))
    else:
        weights = np.array([]).astype(float)

    return weights


def get_weights(weight_type, sub_conf, sub_y, sub_y_unmodified, threshold):
    if weight_type == "confidence":
        sub_idx = sub_y == 0
        sub_weights = copy.deepcopy(sub_conf)
        sub_weights[sub_idx] = 1
        weights = sub_weights
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

        weights = sub_weights
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

        weights = sub_weights
    elif weight_type == "oracle":
        fp_idx = np.logical_and(sub_conf > threshold, sub_y_unmodified == 0)
        sub_weights = np.ones(len(sub_y_unmodified))
        sub_weights[fp_idx] = 0

        weights = sub_weights
    else:
        weights = np.array(np.ones(len(sub_y)))

    return weights


def build_cumulative_data(cumulative_data, cumulative_x_update, cumulative_y_update, sub_x, sub_y):
    if cumulative_data:
        cumulative_x_update = np.concatenate((sub_x, cumulative_x_update))
        cumulative_y_update = np.concatenate((sub_y, cumulative_y_update))

    return cumulative_x_update, cumulative_y_update


def build_cumulative_weights(cumulative_data, cumulative_weights, sub_weights):
    if cumulative_data:
        cumulative_weights = np.concatenate([sub_weights, cumulative_weights])

    return cumulative_weights


def split_validation_data(x_train, y_train, cumulative_x_update,
                          cumulative_y_update, sub_x, sub_y, train_weights,
                          cumulative_update_weights, sub_weights,
                          dynamic_desired_partition, threshold_validation_percentage, include_train):
    if dynamic_desired_partition == "train":
        if include_train:
            x_threshold_set, x_threshold_reset, y_threshold_set, y_threshold_reset, temp_weights, _ = train_test_split(x_train,
                                                                                                                       y_train,
                                                                                                                       train_weights,
                                                                                                                       stratify=y_train,
                                                                                                                       test_size=threshold_validation_percentage)
            all_train_x = np.concatenate([x_threshold_set, cumulative_x_update, sub_x])
            all_train_y = np.concatenate([y_threshold_set, cumulative_y_update, sub_y])
            all_train_weights = np.concatenate([temp_weights, cumulative_update_weights, sub_weights])
            all_valid_x = x_threshold_reset
            all_valid_y = y_threshold_reset
        else:
            all_train_x = np.concatenate([cumulative_x_update, sub_x])
            all_train_y = np.concatenate([cumulative_y_update, sub_y])
            all_train_weights = np.concatenate([cumulative_update_weights, sub_weights])
            all_valid_x = x_train
            all_valid_y = y_train
    elif dynamic_desired_partition == "update_current":
        neg_prop = np.sum(sub_y == 0) / len(sub_y)
        pos_prop = np.sum(sub_y == 1) / len(sub_y)

        if (int(neg_prop * threshold_validation_percentage * len(sub_y)) <= 1 or
            int(pos_prop * threshold_validation_percentage * len(sub_y)) <= 1):
            strat = None
        else:
            strat = sub_y

        x_threshold_set, x_threshold_reset, y_threshold_set, y_threshold_reset, temp_weights, _ = train_test_split(sub_x,
                                                                                                                   sub_y,
                                                                                                                   sub_weights,
                                                                                                                   stratify=strat,
                                                                                                                   test_size=threshold_validation_percentage)
        if include_train:
            all_train_x = np.concatenate([x_train, cumulative_x_update, x_threshold_set])
            all_train_y = np.concatenate([y_train, cumulative_y_update, y_threshold_set])
            all_train_weights = np.concatenate([train_weights, cumulative_update_weights, temp_weights])

        else:
            all_train_x = np.concatenate([cumulative_x_update, x_threshold_set])
            all_train_y = np.concatenate([cumulative_y_update, y_threshold_set])
            all_train_weights = np.concatenate([cumulative_update_weights, temp_weights])

        all_valid_x = x_threshold_reset
        all_valid_y = y_threshold_reset
    elif dynamic_desired_partition == "update_cumulative":
        x_threshold_set, x_threshold_reset, y_threshold_set, y_threshold_reset, temp_weights, _ = train_test_split(np.concatenate([cumulative_x_update, sub_x]),
                                                                                                                   np.concatenate([cumulative_y_update, sub_y]),
                                                                                                                   np.concatenate([cumulative_update_weights, sub_weights]),
                                                                                                                   stratify=np.concatenate([cumulative_y_update, sub_y]),
                                                                                                                   test_size=threshold_validation_percentage)
        if include_train:
            all_train_x = np.concatenate([x_train, x_threshold_set])
            all_train_y = np.concatenate([y_train, y_threshold_set])
            all_train_weights = np.concatenate([train_weights, temp_weights])

        else:
            all_train_x = x_threshold_set
            all_train_y = y_threshold_set
            all_train_weights = temp_weights

        all_valid_x = x_threshold_reset
        all_valid_y = y_threshold_reset
    elif dynamic_desired_partition == "all":
        if include_train:
            x_threshold_set, x_threshold_reset, y_threshold_set, y_threshold_reset, temp_weights, _ = train_test_split(np.concatenate([x_train, cumulative_x_update, sub_x]),
                                                                                                                       np.concatenate([y_train, cumulative_y_update, sub_y]),
                                                                                                                        np.concatenate([train_weights, cumulative_update_weights, sub_weights]),
                                                                                                                       stratify=np.concatenate([y_train, cumulative_y_update, sub_y]),
                                                                                                                       test_size=threshold_validation_percentage)
            all_train_x = x_threshold_set
            all_train_y = y_threshold_set
            all_train_weights = temp_weights
            all_valid_x = x_threshold_reset
            all_valid_y = y_threshold_reset
        else:
            x_threshold_set, x_threshold_reset, y_threshold_set, y_threshold_reset, temp_weights, _ = train_test_split(np.concatenate([cumulative_x_update, sub_x]),
                                                                                                                       np.concatenate([cumulative_y_update, sub_y]),
                                                                                                                        np.concatenate([cumulative_update_weights, sub_weights]),
                                                                                                                       stratify=np.concatenate([cumulative_y_update, sub_y]),
                                                                                                                       test_size=threshold_validation_percentage)
            all_train_x = x_threshold_set
            all_train_y = y_threshold_set
            all_train_weights = temp_weights
            all_valid_x = np.concatenate([x_train, x_threshold_reset])
            all_valid_y = np.concatenate([y_train, y_threshold_reset])

    return all_train_x, all_train_y, all_train_weights, all_valid_x, all_valid_y


def combine_data(x_train, y_train, cumulative_x_update,
                 cumulative_y_update, sub_x, sub_y, train_weights,
                 cumulative_update_weights, sub_weights, include_train):
    if include_train:
        all_x = np.concatenate([x_train, cumulative_x_update, sub_x])
        all_y = np.concatenate([y_train, cumulative_y_update, sub_y])
        all_weights = np.concatenate([train_weights, cumulative_update_weights, sub_weights])
    else:
        all_x = np.concatenate([cumulative_x_update, sub_x])
        all_y = np.concatenate([cumulative_y_update, sub_y])
        all_weights = np.concatenate([cumulative_update_weights, sub_weights])

    return all_x, all_y, all_weights