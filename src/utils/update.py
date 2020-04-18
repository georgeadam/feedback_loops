import copy
import numpy as np

from src.utils.label_flip import flip_labels
from src.utils.misc import create_empty_rates
from src.utils.metrics import compute_all_rates
from src.utils.sample_reweighting import get_weights
from src.utils.threshold import find_threshold
from src.utils.trust import full_trust, conditional_trust, constant_trust

from sklearn.model_selection import train_test_split


def wrapped(fn, **kwargs):
    def inside(*args, **specified_args):
        return fn(*args, **kwargs, **specified_args)

    return inside


def get_update_fn(update_type, temporal=False):
    if temporal:
        update_fn = update_model_temporal
    else:
        update_fn = update_model_static

    if update_type == "feedback_online_single_batch":
        return wrapped(update_fn, agg_data=False, include_train=False, weight_type=None,
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_online_all_update_data":
        return wrapped(update_fn, agg_data=True, include_train=False, weight_type=None,
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_online_all_data":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_full_fit" or update_type == "feedback_full_fit_cad":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
                fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_past_year" or update_type == "feedback_full_fit_past_year_cad":
        return wrapped(update_fn, agg_data=False, include_train=False, weight_type=None,
                fit_type="fit", feedback=True)
    elif update_type == "feedback_online_all_update_data_weighted":
        return wrapped(update_fn, agg_data=True, include_train=False, weight_type=None,
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_online_all_data_weighted":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="linearly_decreasing",
                       fit_type="partial_fit", feedback=True)
    elif update_type == "feedback_full_fit_weighted":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="linearly_decreasing",
                fit_type="fit", feedback=True)
    elif update_type == "no_feedback_online_single_batch":
        return wrapped(update_fn, agg_data=False, include_train=False, weight_type=None,
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_online_all_update_data":
        return wrapped(update_fn, agg_data=True, include_train=False, weight_type=None,
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_online_all_data":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_full_fit":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
                fit_type="fit", feedback=False)
    elif update_type == "no_feedback_full_fit_past_year":
        return wrapped(update_fn, agg_data=False, include_train=False, weight_type=None,
                fit_type="fit", feedback=False)
    elif update_type == "no_feedback_online_all_update_data_weighted":
        return wrapped(update_fn, agg_data=True, include_train=False, weight_type="linearly_decreasing",
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_online_all_data_weighted":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="linearly_decreasing",
                       fit_type="partial_fit", feedback=False)
    elif update_type == "no_feedback_full_fit_weighted":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="linearly_decreasing",
                fit_type="fit", feedback=False)
    elif update_type == "feedback_full_fit_confidence":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="confidence",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_drop":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_drop_low_confidence":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop_low_confidence",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_drop_random":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="random",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_drop_everything":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop_everything",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_flip_everything":
        return wrapped(update_fn, agg_data=True, include_train=True,
                       fit_type="fit", feedback=True, flip_type="flip_everything")
    elif update_type == "feedback_full_fit_flip_oracle":
        return wrapped(update_fn, agg_data=True, include_train=True,
                       fit_type="fit", feedback=True, flip_type="oracle")
    elif update_type == "feedback_full_fit_flip_random":
        return wrapped(update_fn, agg_data=True, include_train=True,
                       fit_type="fit", feedback=True, flip_type="random")
    elif update_type == "feedback_full_fit_flip_low_confidence":
        return wrapped(update_fn, agg_data=True, include_train=True,
                       fit_type="fit", feedback=True, flip_type="flip_low_confidence")
    elif update_type == "feedback_full_fit_oracle":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="oracle",
                       fit_type="fit", feedback=True)
    elif update_type == "feedback_full_fit_conditional_trust":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
                fit_type="fit", feedback=True, trust_fn=conditional_trust)
    elif update_type == "feedback_full_fit_constant_trust":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
                fit_type="fit", feedback=True, trust_fn=constant_trust)
    elif update_type == "no_feedback_full_fit_confidence":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="confidence",
                       fit_type="fit", feedback=False)
    elif update_type == "no_feedback_full_fit_drop":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop",
                       fit_type="fit", feedback=False)
    elif update_type == "no_feedback_full_fit_drop_low_confidence":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop_low_confidence",
                       fit_type="fit", feedback=False)
    elif update_type == "no_feedback_full_fit_oracle":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
                       fit_type="fit", feedback=False)
    elif update_type == "evaluate":
        return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
                       fit_type="fit", feedback=False, update=False)


def map_update_type(update_type):
    if update_type.startswith("feedback_online_single_batch"):
        return "ssrd"
    elif update_type.startswith("no_feedback_full_fit_confidence"):
        return "no_feedback_confidence"
    elif update_type.startswith("no_feedback_full_fit_drop"):
        return "no_feedback_drop"
    elif update_type.startswith("feedback_full_fit_cad"):
        return "refit_all_data"
    elif update_type.startswith("feedback_full_fit_flip_everything"):
        return "feedback_flip_all_pos"
    elif update_type.startswith("feedback_full_fit_flip_oracle"):
        return "feedback_flip_oracle"
    elif update_type.startswith("feedback_full_fit_flip_random"):
        return "feedback_flip_random"
    elif update_type.startswith("feedback_full_fit_flip_low_confidence"):
        return "feedback_flip_low_confidence"
    elif update_type.startswith("feedback_full_fit_past_year_cad"):
        return "refit_past_year"
    elif update_type.startswith("feedback_full_fit_drop_everything"):
        return "feedback_drop_all_pos"
    elif update_type.startswith("no_feedback_full_fit_drop_low_confidence"):
        return "no_feedback_drop_low_confidence"
    elif update_type.startswith("no_feedback_full_fit_oracle"):
        return "no_feedback_oracle"
    elif update_type.startswith("no_feedback_full_fit_past_year"):
        return "no_feedback_past_year"
    elif update_type.startswith("feedback_full_fit_confidence"):
        return "feedback_confidence"
    elif update_type.startswith("feedback_full_fit_drop_random"):
        return "feedback_drop_random"
    elif update_type.startswith("feedback_full_fit_drop_low_confidence"):
        return "feedback_drop_low_confidence"
    elif update_type.startswith("feedback_full_fit_past_year"):
        return "feedback_past_year"
    elif update_type.startswith("feedback_online_all_update_data"):
        return "ssad-nt"
    elif update_type.startswith("feedback_online_all_data"):
        return "ssad-t"
    elif update_type.startswith("feedback_full_fit_conditional_trust"):
        return "feedback_conditional_trust"
    elif update_type.startswith("feedback_full_fit_constant_trust"):
        return "feedback_constant_trust"
    elif update_type.startswith("feedback_full_fit_oracle"):
        return "feedback_oracle"
    elif update_type.startswith("feedback_full_fit"):
        return "feedback_all_data"
    elif update_type.startswith("no_feedback"):
        return "no_feedback_all_data"
    elif update_type.startswith("evaluate"):
        return "static"


def update_model_static(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                        agg_data=False, include_train=False, weight_type=None, fit_type="fit", feedback=False,
                        update=True, intermediate=False, trust_fn=full_trust, clinician_fpr=0.0, clinician_trust=1.0,
                        threshold=None, ddr=None, ddv=None, ddp=None, tvp=0.2, scaler=None, flip_type=None):
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)
    agg_x_update = np.array([]).astype(float).reshape(0, x_train.shape[1])
    agg_y_update = np.array([]).astype(int)

    rates = create_empty_rates()
    train_weights = initialize_weights(weight_type, x_train, include_train)
    agg_update_weights = np.array([]).astype(float)

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])
        sub_y_unmodified = copy.deepcopy(y_update[idx_start: idx_end])
        sub_conf = new_model.predict_proba(scaler.transform(sub_x))[:, 1]

        if i == 0:
            initial_fpr = compute_initial_fpr(new_model, threshold, sub_x, sub_y, scaler)
            model_fpr = initial_fpr
        else:
            model_fpr = compute_model_fpr(new_model, x_train, y_train, threshold, scaler)

        sub_y = replace_labels(feedback, new_model, sub_x, sub_y, threshold, trust_fn, clinician_fpr, clinician_trust,
                               model_fpr, scaler)
        sub_y = flip_labels(flip_type, sub_conf, sub_y, sub_y_unmodified, threshold)
        sub_weights = get_weights(weight_type, sub_conf, sub_y, sub_y_unmodified, threshold)
        update_scaler(x_train, agg_x_update, sub_x, include_train, scaler)
        threshold = make_update(x_train, y_train, agg_x_update, agg_y_update, sub_x, sub_y, train_weights,
                                agg_update_weights, sub_weights, new_model, threshold, fit_type, update, include_train,
                                ddr, ddv, ddp, tvp, agg_data, scaler)

        loss = new_model.evaluate(x_test, y_test)
        append_rates(intermediate, new_model, rates, threshold, x_test, y_test, scaler)
        rates["loss"][-1] = loss
        agg_x_update, agg_y_update = build_agg_data(agg_data, agg_x_update,
                                                    agg_y_update, sub_x, sub_y)
        agg_update_weights = build_agg_weights(agg_data, agg_update_weights, sub_weights)

    return new_model, rates


def update_model_temporal(model, x_train, y_train, x_rest, y_rest, years, tyl=1999, uyl=2019, agg_data=False,
                          include_train=False, weight_type=None, fit_type="fit", feedback=False, update=True,
                          next_year=True, trust_fn=full_trust, clinician_fpr=0.0, clinician_trust=1.0,
                          intermediate=False, threshold=None, ddr=None, ddv=None, ddp=None, tvp=0.2, scaler=None,
                          flip_type=None):
    new_model = copy.deepcopy(model)

    agg_x_update = np.array([]).astype(float).reshape(0, x_train.shape[1])
    agg_y_update = np.array([]).astype(int)

    rates = create_empty_rates()

    train_weights = initialize_weights(weight_type, x_train, include_train)
    agg_update_weights = np.array([]).astype(float)

    for year in range(tyl + 1, uyl):
        sub_idx = years == year

        if next_year:
            test_idx = years == year + 1
        else:
            test_idx = years == uyl

        sub_x = x_rest[sub_idx]
        sub_y = copy.deepcopy(y_rest[sub_idx])
        sub_y_unmodified = copy.deepcopy(y_rest[sub_idx])
        temp_conf = new_model.predict_proba(scaler.transform(sub_x))

        if temp_conf.shape[1] > 1:
            sub_conf = temp_conf[:, 1]
        else:
            sub_conf = temp_conf[:, 0]

        if year == tyl + 1:
            initial_fpr = compute_initial_fpr(model, threshold, x_rest[sub_idx], y_rest[sub_idx], scaler)
            model_fpr = initial_fpr
        else:
            model_fpr = compute_model_fpr(new_model, x_train, y_train, threshold, scaler)
            # model_fpr = rates["fpr"][-1]
        sub_y = replace_labels(feedback, new_model, sub_x, sub_y, threshold, trust_fn, clinician_fpr,
                               clinician_trust, model_fpr, scaler)
        sub_y = flip_labels(flip_type, sub_conf, sub_y, sub_y_unmodified, threshold)
        sub_weights = get_weights(weight_type, sub_conf, sub_y, sub_y_unmodified, threshold)
        update_scaler(x_train, agg_x_update, sub_x, include_train, scaler)
        threshold = make_update(x_train, y_train, agg_x_update, agg_y_update, sub_x, sub_y, train_weights,
                                agg_update_weights, sub_weights, new_model, threshold, fit_type, update, include_train,
                                ddr, ddv, ddp, tvp, agg_data, scaler)

        loss = new_model.evaluate(scaler.transform(x_rest[test_idx]), y_rest[test_idx])
        append_rates(intermediate, new_model, rates, threshold, x_rest[test_idx], y_rest[test_idx], scaler)
        rates["loss"][-1] = (loss)

        agg_x_update, agg_y_update = build_agg_data(agg_data, agg_x_update,
                                                    agg_y_update, sub_x, sub_y)
        agg_update_weights = build_agg_weights(agg_data, agg_update_weights, sub_weights)


    return new_model, rates


def compute_initial_fpr(model, threshold, x_train, y_train, scaler):
    temp_prob = model.predict_proba(scaler.transform(x_train))[:, 1]
    temp_pred = temp_prob > threshold
    initial_fps = np.logical_and(temp_pred == 1, y_train == 0)
    initial_fpr = len(y_train[initial_fps]) / len(y_train)

    return initial_fpr


def make_update(x_train, y_train, agg_x_update, agg_y_update, sub_x, sub_y, train_weights,
                agg_update_weights, sub_weights, new_model, threshold, fit_type, update, include_train,
                ddr, ddv, ddp, tvp, agg_data, scaler):
    if update:
        if ddr is not None:
            all_train_x, all_train_y, all_train_weights, \
            all_valid_x, all_valid_y, all_valid_weights = split_validation_data(x_train, y_train, agg_x_update,
                                                                                agg_y_update, sub_x, sub_y, train_weights,
                                                                                agg_update_weights, sub_weights, ddp,
                                                                                tvp, include_train, agg_data)

            if fit_type == "partial_fit":
                getattr(new_model, fit_type)(scaler.transform(all_train_x), all_train_y, classes=np.array([0, 1]),
                                             sample_weight=all_train_weights)
            else:
                getattr(new_model, fit_type)(scaler.transform(all_train_x), all_train_y,
                                             sample_weight=all_train_weights)
            valid_prob = new_model.predict_proba(scaler.transform(all_valid_x))

            temp_idx = all_valid_weights !=0
            prev_threshold = threshold
            threshold = find_threshold(all_valid_y[temp_idx], valid_prob[temp_idx], ddr, ddv)

            if threshold is None:
                threshold = prev_threshold

        all_x, all_y, all_weights = combine_data(x_train, y_train, agg_x_update,
                                                 agg_y_update, sub_x, sub_y, train_weights,
                                                 agg_update_weights, sub_weights, include_train)
        if fit_type == "partial_fit":
            getattr(new_model, fit_type)(scaler.transform(all_x), all_y, classes=np.array([0, 1]),
                                         sample_weight=all_weights)
        else:
            getattr(new_model, fit_type)(scaler.transform(all_x), all_y, sample_weight=all_weights)

    return threshold


def replace_labels(feedback, new_model, sub_x, sub_y, threshold, trust_fn=None, clinician_fpr=0.0, clinician_trust=1.0,
                   model_fpr=0.2, scaler=None):
    if feedback:
        model_pred = new_model.predict_proba(scaler.transform(sub_x))
        if model_pred.shape[1] > 1:
            model_pred = model_pred[:, 1] > threshold
        else:
            model_pred = model_pred[:, 0] > threshold

        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1
    else:
        model_pred = copy.deepcopy(sub_y)


    trust = trust_fn(model_fpr=model_fpr, clinician_trust=clinician_trust)

    clinician_pred = copy.deepcopy(sub_y)
    neg_idx = np.where(clinician_pred == 0)[0]
    physician_fp_idx = np.random.choice(neg_idx, min(int(clinician_fpr * len(sub_y)), len(neg_idx)))
    clinician_pred[physician_fp_idx] = 1

    bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])

    target = bernoulli * model_pred + (1 - bernoulli) * clinician_pred

    return target


def append_rates(intermediate, new_model, rates, threshold, x_test, y_test, scaler):
    if intermediate:
        if threshold is None:
            pred_prob = new_model.predict_proba(scaler.transform(x_test))
            new_pred = new_model.predict(scaler.transform(x_test))
        else:
            pred_prob = new_model.predict_proba(scaler.transform(x_test))

            if pred_prob.shape[1] > 1:
                new_pred = pred_prob[:, 1] >= threshold
            else:
                new_pred = pred_prob[:, 0] >= threshold

        updated_rates = compute_all_rates(y_test, new_pred, pred_prob)

        for key in rates.keys():
            if key in updated_rates.keys():
                if (key == "fp_count" or key == "total_samples") and len(rates[key]) > 0:
                    rates[key].append(rates[key][-1] + updated_rates[key])
                else:
                    rates[key].append(updated_rates[key])

        rates["fp_prop"].append(rates["fp_count"][-1] / rates["total_samples"][-1])


def initialize_weights(weight_type, x_train, include_train):
    if weight_type is None:
        weights = np.array(np.ones(len(x_train)))
    elif include_train:
        weights = np.array(np.ones(len(x_train)))
    else:
        weights = np.array([]).astype(float)

    return weights


def build_agg_data(agg_data, agg_x_update, agg_y_update, sub_x, sub_y):
    if agg_data:
        agg_x_update = np.concatenate((sub_x, agg_x_update))
        agg_y_update = np.concatenate((sub_y, agg_y_update))

    return agg_x_update, agg_y_update


def build_agg_weights(agg_data, agg_weights, sub_weights):
    if agg_data:
        agg_weights = np.concatenate([sub_weights, agg_weights])

    return agg_weights


def split_validation_data(x_train, y_train, agg_x_update, agg_y_update, sub_x, sub_y, train_weights, agg_update_weights,
                          sub_weights, ddp, tvp, include_train, agg_data):
    if ddp == "train":
        if include_train:
            x_threshold_set, x_threshold_reset, \
            y_threshold_set, y_threshold_reset, weights_set, weights_reset = train_test_split(x_train, y_train,
                                                                                              train_weights,
                                                                                              stratify=y_train,
                                                                                              test_size=tvp)
            all_train_x = np.concatenate([x_threshold_set, agg_x_update, sub_x])
            all_train_y = np.concatenate([y_threshold_set, agg_y_update, sub_y])
            all_train_weights = np.concatenate([weights_set, agg_update_weights, sub_weights])
            all_valid_x = x_threshold_reset
            all_valid_y = y_threshold_reset
            all_valid_weights = weights_reset
        else:
            all_train_x = np.concatenate([agg_x_update, sub_x])
            all_train_y = np.concatenate([agg_y_update, sub_y])
            all_train_weights = np.concatenate([agg_update_weights, sub_weights])
            all_valid_x = x_train
            all_valid_y = y_train
            all_valid_weights = train_weights
    elif ddp == "update_current":
        neg_prop = np.sum(sub_y == 0) / len(sub_y)
        pos_prop = np.sum(sub_y == 1) / len(sub_y)

        if (int(neg_prop * tvp * len(sub_y)) <= 1 or
            int(pos_prop * tvp * len(sub_y)) <= 1):
            strat = None
        else:
            strat = sub_y

        x_threshold_set, x_threshold_reset, \
        y_threshold_set, y_threshold_reset, weights_set, weights_reset = train_test_split(sub_x, sub_y, sub_weights,
                                                                                          stratify=strat, test_size=tvp)
        if include_train:
            all_train_x = np.concatenate([x_train, agg_x_update, x_threshold_set])
            all_train_y = np.concatenate([y_train, agg_y_update, y_threshold_set])
            all_train_weights = np.concatenate([train_weights, agg_update_weights, weights_set])
        else:
            all_train_x = np.concatenate([agg_x_update, x_threshold_set])
            all_train_y = np.concatenate([agg_y_update, y_threshold_set])
            all_train_weights = np.concatenate([agg_update_weights, weights_set])

        all_valid_x = x_threshold_reset
        all_valid_y = y_threshold_reset
        all_valid_weights = weights_reset
    elif ddp == "update_cumulative":
        temp_x = np.concatenate([agg_x_update, sub_x])
        temp_y = np.concatenate([agg_y_update, sub_y])
        temp_weights = np.concatenate([agg_update_weights, sub_weights])
        x_threshold_set, x_threshold_reset, \
        y_threshold_set, y_threshold_reset, weights_set, weights_reset = train_test_split(temp_x, temp_y, temp_weights,
                                                                                          stratify=temp_y, test_size=tvp)
        if include_train:
            all_train_x = np.concatenate([x_train, x_threshold_set])
            all_train_y = np.concatenate([y_train, y_threshold_set])
            all_train_weights = np.concatenate([train_weights, weights_set])

        else:
            all_train_x = x_threshold_set
            all_train_y = y_threshold_set
            all_train_weights = weights_set

        all_valid_x = x_threshold_reset
        all_valid_y = y_threshold_reset
        all_valid_weights = weights_reset
    elif ddp == "all":
        if include_train:
            temp_x = np.concatenate([x_train, agg_x_update, sub_x])
            temp_y = np.concatenate([y_train, agg_y_update, sub_y])
            temp_weights = np.concatenate([train_weights, agg_update_weights, sub_weights])
            x_threshold_set, x_threshold_reset, \
            y_threshold_set, y_threshold_reset, weights_set, weights_reset = train_test_split(temp_x, temp_y, temp_weights,
                                                                                              stratify=temp_y, test_size=tvp)
            all_train_x = x_threshold_set
            all_train_y = y_threshold_set
            all_train_weights = weights_set
            all_valid_x = x_threshold_reset
            all_valid_y = y_threshold_reset
            all_valid_weights = weights_reset
        else:
            neg_prop = (np.sum(sub_y == 0) + np.sum(agg_y_update) == 0) / len(sub_y)
            pos_prop = (np.sum(sub_y == 1) + np.sum(agg_y_update == 1)) / len(sub_y)

            if (int(neg_prop * tvp * len(sub_y)) <= 1 or
                    int(pos_prop * tvp * len(sub_y)) <= 1):
                strat = None
            else:
                strat = np.concatenate([agg_y_update, sub_y])

            temp_x = np.concatenate([agg_x_update, sub_x])
            temp_y = np.concatenate([agg_y_update, sub_y])
            temp_weights = np.concatenate([agg_update_weights, sub_weights])
            x_threshold_set, x_threshold_reset, \
            y_threshold_set, y_threshold_reset, weights_set, weights_reset = train_test_split(temp_x, temp_y, temp_weights,
                                                                                              stratify=strat, test_size=tvp)
            all_train_x = x_threshold_set
            all_train_y = y_threshold_set
            all_train_weights = weights_set

            if agg_data:
                # all_valid_x = np.concatenate([x_train, x_threshold_reset])
                # all_valid_y = np.concatenate([y_train, y_threshold_reset])
                all_valid_x = np.concatenate([x_threshold_reset])
                all_valid_y = np.concatenate([y_threshold_reset])
                all_valid_weights = weights_reset
            else:
                all_valid_x = np.concatenate([x_threshold_reset])
                all_valid_y = np.concatenate([y_threshold_reset])
                all_valid_weights = weights_reset

    return all_train_x, all_train_y, all_train_weights, all_valid_x, all_valid_y, all_valid_weights


def combine_data(x_train, y_train, agg_x_update, agg_y_update, sub_x, sub_y, train_weights, agg_update_weights,
                 sub_weights, include_train):
    if include_train:
        all_x = np.concatenate([x_train, agg_x_update, sub_x])
        all_y = np.concatenate([y_train, agg_y_update, sub_y])
        all_weights = np.concatenate([train_weights, agg_update_weights, sub_weights])
    else:
        all_x = np.concatenate([agg_x_update, sub_x])
        all_y = np.concatenate([agg_y_update, sub_y])
        all_weights = np.concatenate([agg_update_weights, sub_weights])

    return all_x, all_y, all_weights


def update_scaler(x_train, agg_x_update, sub_x, include_train, scaler):
    if include_train:
        all_x = np.concatenate([x_train, agg_x_update, sub_x])
    else:
        all_x = np.concatenate([agg_x_update, sub_x])


    scaler.fit(all_x)


def compute_model_fpr(model, x, y, threshold, scaler):
    y_prob = model.predict_proba(scaler.transform(x))
    y_pred = y_prob[:, 1] > threshold

    fp_idx = np.logical_and(y == 0, y_pred == 1)
    neg = np.sum(y == 0)

    return float(np.sum(fp_idx) / neg)