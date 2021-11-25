import copy
import numpy as np

from src.utils.metrics import compute_model_fpr
from src.utils.threshold import find_threshold
from src.utils.trust import full_trust, get_trust_fn

from sklearn.model_selection import train_test_split
from src.utils.typing import Model, Transformer
from typing import Callable, Optional


def wrapped(fn, **kwargs):
    def inside(*args, **specified_args):
        return fn(*args, **kwargs, **specified_args)

    return inside


def get_update_fn(args):
    update_fn = update_model_general

    return wrapped(update_fn,
                   feedback=args.update_params.feedback, trust_fn=get_trust_fn(args.trust.type),
                   update=args.update_params.do_update)


def update_model_general(model, data_wrapper, rate_tracker, trainer=None,
                         feedback: bool = False,
                         update: bool = True, trust_fn: Callable = full_trust, clinician_fpr: float = 0.0,
                         clinician_trust: float = 1.0,
                         ddv: Optional[float] = None, scaler: Transformer = None,
                         recover_prob: float = 1.0):
    x_train, y_train = data_wrapper.get_train_data()

    for update_num, (x_update, y_update) in enumerate(data_wrapper.get_update_data_generator(), start=1):
        # Create copies of update labels so we don't overwrite original data
        y_update = copy.deepcopy(y_update)
        y_update_unmodified = copy.deepcopy(y_update)

        # Compute FPR on update data to be used by conditional clinician trust
        model_fpr = compute_model_fpr(model, x_update, y_update, scaler)
        y_update = replace_labels(feedback, model, x_update, y_update, trust_fn,
                                  clinician_fpr, clinician_trust, model_fpr, recover_prob, scaler)

        data_wrapper.store_current_update_batch_corrupt(x_update, y_update)
        data_wrapper.store_current_update_batch_clean(x_update, y_update_unmodified)

        # Update data normalization to take into account new data
        refit_scaler(scaler, data_wrapper, update)
        model = trainer.update_fit(model, data_wrapper, rate_tracker, scaler, update_num)

        model.threshold = refit_threshold(model, data_wrapper, update, ddv, scaler)

        x_eval, y_eval = data_wrapper.get_eval_data(update_num)
        track_performance(model, rate_tracker, x_eval, y_eval, scaler)

        data_wrapper.accumulate_update_data()
        rate_tracker.update_detection(y_train, data_wrapper.get_cumulative_update_data()[1])

    return model


def refit_threshold(model: Model, data_wrapper, update: bool, ddv: float, scaler: Transformer):
    ddr = data_wrapper.get_ddr()
    if update:
        if ddr is not None:
            all_thresh_x, all_thresh_y = data_wrapper.get_all_data_for_threshold_fit()
            valid_prob = model.predict_proba(scaler.transform(all_thresh_x))

            prev_threshold = model.threshold
            new_threshold = find_threshold(all_thresh_y, valid_prob, ddr, ddv)

            if new_threshold is None:
                return prev_threshold
        else:
            return model.threshold

    return model.threshold


def replace_labels(feedback: str, new_model: Model, x: np.ndarray, y: np.ndarray,
                   trust_fn: Callable = None, clinician_fpr: float = 0.0, clinician_trust: float = 1.0,
                   model_fpr: float = 0.2, recover_prob: float= 1.0, scaler: Transformer = None):
    if (feedback == "amplify") or (feedback is True):
        model_pred, model_prob = replace_label_bias_amplification(new_model, scaler, new_model.threshold, x, y)
    elif feedback == "oscillate":
        model_pred, model_prob = replace_label_bias_oscillation(new_model, scaler, new_model.threshold, recover_prob, x, y)
    else:
        model_pred, model_prob = compute_model_pred(new_model, scaler, x, y)

    trust = trust_fn(model_fpr=model_fpr, model_prob=model_prob, clinician_trust=clinician_trust)

    clinician_pred = copy.deepcopy(y)
    neg_idx = np.where(clinician_pred == 0)[0]
    physician_fp_idx = np.random.choice(neg_idx, min(int(clinician_fpr * len(y)), len(neg_idx)))
    clinician_pred[physician_fp_idx] = 1

    if type(trust) != float:
        bernoulli = np.random.binomial([1] * len(y), trust)
    else:
        bernoulli = np.random.choice([0, 1], len(y), p=[1 - trust, trust])

    target = bernoulli * model_pred + (1 - bernoulli) * clinician_pred

    return target


def compute_model_pred(new_model, scaler, x, y):
    model_prob = new_model.predict_proba(scaler.transform(x))
    model_pred = copy.deepcopy(y)
    model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]

    return model_pred, model_prob


def replace_label_bias_oscillation(new_model, scaler, threshold, recover_prob, x, y):
    model_prob = new_model.predict_proba(scaler.transform(x))

    if model_prob.shape[1] > 1:
        model_pred = model_prob[:, 1] > threshold
    else:
        model_pred = model_prob[:, 0] > threshold

    model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]
    model_tp_idx = np.where(np.logical_and(y == 1, model_pred == 1))[0]
    idx = np.random.choice(model_tp_idx, int(recover_prob * len(model_tp_idx)))
    model_pred = copy.deepcopy(y)
    model_pred[idx] = 0

    return model_pred, model_prob


def replace_label_bias_amplification(new_model, scaler, threshold, x, y):
    model_prob = new_model.predict_proba(scaler.transform(x))

    if model_prob.shape[1] > 1:
        model_pred = model_prob[:, 1] > threshold
    else:
        model_pred = model_prob[:, 0] > threshold

    model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]
    model_fp_idx = np.where(np.logical_and(y == 0, model_pred == 1))[0]
    model_pred = copy.deepcopy(y)
    model_pred[model_fp_idx] = 1

    return model_pred, model_prob


def track_performance(new_model: Model, rate_tracker, x_eval: np.ndarray, y_eval: np.ndarray, scaler: Transformer):
    if new_model.threshold is None:
        eval_prob = new_model.predict_proba(scaler.transform(x_eval))
        eval_pred = new_model.predict(scaler.transform(x_eval))
    else:
        eval_prob = new_model.predict_proba(scaler.transform(x_eval))

        if eval_prob.shape[1] > 1:
            eval_pred = eval_prob[:, 1] >= new_model.threshold
        else:
            eval_pred = eval_prob[:, 0] >= new_model.threshold

    rate_tracker.update_rates(y_eval, eval_pred, eval_prob)


def refit_scaler(scaler: Transformer, data_wrapper, update: bool):
    if update:
        x = data_wrapper.get_all_data_for_scaler_fit()

        scaler.fit(x)
