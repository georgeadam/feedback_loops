import copy
import numpy as np

from src.utils.detection import check_feedback_loop
from src.utils.label_flip import flip_labels
from src.utils.misc import create_empty_rates
from src.utils.metrics import compute_model_fpr
from src.utils.sample_reweighting import get_weights, initialize_weights, build_cumulative_weights
from src.utils.threshold import find_threshold
from src.utils.trust import full_trust, conditional_trust, constant_trust, confidence_trust, confidence_threshold_trust, get_trust_fn

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


# def get_update_fn(update_type: str):
#     update_fn = update_model_general
#
#     if update_type == "feedback_online_single_batch":
#         return wrapped(update_fn, agg_data=False, include_train=False, weight_type=None,
#                        fit_type="partial_fit", feedback=True)
#     elif update_type == "feedback_online_all_update_data":
#         return wrapped(update_fn, agg_data=True, include_train=False, weight_type=None,
#                        fit_type="partial_fit", feedback=True)
#     elif update_type == "feedback_online_all_data":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="partial_fit", feedback=True)
#     elif update_type == "feedback_full_fit" or update_type == "feedback_full_fit_cad":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_full_fit_past_year" or update_type == "feedback_full_fit_past_year_cad":
#         return wrapped(update_fn, agg_data=False, include_train=False, weight_type=None,
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_online_all_update_data_weighted":
#         return wrapped(update_fn, agg_data=True, include_train=False, weight_type=None,
#                        fit_type="partial_fit", feedback=True)
#     elif update_type == "feedback_online_all_data_weighted":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="linearly_decreasing",
#                        fit_type="partial_fit", feedback=True)
#     elif update_type == "feedback_full_fit_weighted":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="linearly_decreasing",
#                        fit_type="fit", feedback=True)
#     elif update_type == "no_feedback_online_single_batch":
#         return wrapped(update_fn, agg_data=False, include_train=False, weight_type=None,
#                        fit_type="partial_fit", feedback=False)
#     elif update_type == "no_feedback_online_all_update_data":
#         return wrapped(update_fn, agg_data=True, include_train=False, weight_type=None,
#                        fit_type="partial_fit", feedback=False)
#     elif update_type == "no_feedback_online_all_data":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="partial_fit", feedback=False)
#     elif update_type == "no_feedback_full_fit":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="fit", feedback=False)
#     elif update_type == "no_feedback_full_fit_past_year":
#         return wrapped(update_fn, agg_data=False, include_train=False, weight_type=None,
#                        fit_type="fit", feedback=False)
#     elif update_type == "no_feedback_online_all_update_data_weighted":
#         return wrapped(update_fn, agg_data=True, include_train=False, weight_type="linearly_decreasing",
#                        fit_type="partial_fit", feedback=False)
#     elif update_type == "no_feedback_online_all_data_weighted":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="linearly_decreasing",
#                        fit_type="partial_fit", feedback=False)
#     elif update_type == "no_feedback_full_fit_weighted":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="linearly_decreasing",
#                        fit_type="fit", feedback=False)
#     elif update_type == "feedback_full_fit_confidence":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="confidence",
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_full_fit_drop":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop",
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_full_fit_drop_low_confidence":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop_low_confidence",
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_full_fit_drop_random":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="random",
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_full_fit_drop_everything":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop_everything",
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_full_fit_drop_all_pos":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop_all_pos",
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_full_fit_flip_everything":
#         return wrapped(update_fn, agg_data=True, include_train=True,
#                        fit_type="fit", feedback=True, flip_type="flip_everything")
#     elif update_type == "feedback_full_fit_flip_oracle":
#         return wrapped(update_fn, agg_data=True, include_train=True,
#                        fit_type="fit", feedback=True, flip_type="oracle")
#     elif update_type == "feedback_full_fit_flip_random":
#         return wrapped(update_fn, agg_data=True, include_train=True,
#                        fit_type="fit", feedback=True, flip_type="random")
#     elif update_type == "feedback_full_fit_flip_low_confidence":
#         return wrapped(update_fn, agg_data=True, include_train=True,
#                        fit_type="fit", feedback=True, flip_type="flip_low_confidence")
#     elif update_type == "feedback_full_fit_oracle":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="oracle",
#                        fit_type="fit", feedback=True)
#     elif update_type == "feedback_full_fit_conditional_trust":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="fit", feedback=True, trust_fn=conditional_trust)
#     elif update_type == "feedback_full_fit_confidence_trust":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="fit", feedback=True, trust_fn=confidence_trust)
#     elif update_type == "feedback_full_fit_confidence_threshold_trust":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="fit", feedback=True, trust_fn=confidence_threshold_trust)
#     elif update_type == "feedback_full_fit_constant_trust":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="fit", feedback=True, trust_fn=constant_trust)
#     elif update_type == "no_feedback_full_fit_confidence":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="confidence",
#                        fit_type="fit", feedback=False)
#     elif update_type == "no_feedback_full_fit_drop":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop",
#                        fit_type="fit", feedback=False)
#     elif update_type == "no_feedback_full_fit_drop_low_confidence":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type="drop_low_confidence",
#                        fit_type="fit", feedback=False)
#     elif update_type == "no_feedback_full_fit_oracle":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="fit", feedback=False)
#     elif update_type == "evaluate":
#         return wrapped(update_fn, agg_data=True, include_train=True, weight_type=None,
#                        fit_type="fit", feedback=False, update=False)


def map_update_type(update_type: str):
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
        return "feedback_drop_all_pos_pred"
    elif update_type.startswith("feedback_full_fit_drop_all_pos"):
        return "feedback_drop_all_pos_label"
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
    elif update_type.startswith("feedback_full_fit_confidence_trust"):
        return "feedback_confidence_trust"
    elif update_type.startswith("feedback_full_fit_confidence_threshold_trust"):
        return "feedback_confidence_threshold_trust"
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


def update_model_general(model, data_wrapper, rate_tracker, trainer=None,
                         feedback: bool = False,
                         update: bool = True, trust_fn: Callable = full_trust, clinician_fpr: float = 0.0,
                         clinician_trust: float = 1.0,
                         threshold: Optional[float] = None, ddv: Optional[float] = None, scaler: Transformer = None,
                         train_lambda: float = 1.0):
    x_train, y_train = data_wrapper.get_train_data()

    for update_num, (x_update, y_update) in enumerate(data_wrapper.get_update_data_generator(), start=1):
        y_update = copy.deepcopy(y_update)
        y_update_unmodified = copy.deepcopy(y_update)

        update_conf = model.predict_proba(scaler.transform(x_update))[:, 1]

        model_fpr = compute_model_fpr(model, x_update, y_update, threshold, scaler)
        y_update = replace_labels(feedback, model, x_update, y_update, threshold, trust_fn,
                                  clinician_fpr, clinician_trust, model_fpr, scaler)

        data_wrapper.store_current_update_batch(x_update, y_update)
        # y_update = flip_labels(flip_type, sub_conf, sub_y, sub_y_unmodified, threshold)
        # No longer interested in effect of flipping labels

        refit_scaler(scaler, data_wrapper, update)
        model = trainer.update_fit(model, data_wrapper, rate_tracker, scaler, update_num)

        threshold = refit_threshold(model, data_wrapper, threshold, update, ddv, scaler)

        x_eval, y_eval = data_wrapper.get_eval_data(update_num)
        track_performance(model, rate_tracker, x_eval, y_eval, threshold, scaler)

        data_wrapper.accumulate_update_data()
        rate_tracker.update_detection(y_train, data_wrapper.get_cumulative_update_data()[1])

    return model


def refit_threshold(model: Model, data_wrapper, threshold: float,
                    update: bool, ddv: float, scaler: Transformer):
    ddr = data_wrapper.get_ddr()
    if update:
        if ddr is not None:
            all_thresh_x, all_thresh_y = data_wrapper.get_all_data_for_threshold_fit()
            valid_prob = model.predict_proba(scaler.transform(all_thresh_x))

            prev_threshold = threshold
            threshold = find_threshold(all_thresh_y, valid_prob, ddr, ddv)

            if threshold is None:
                threshold = prev_threshold
        else:
            return threshold

    return threshold


def replace_labels(feedback: bool, new_model: Model, x: np.ndarray, y: np.ndarray, threshold: float,
                   trust_fn: Callable = None, clinician_fpr: float = 0.0, clinician_trust: float = 1.0,
                   model_fpr: float = 0.2, scaler: Transformer = None):
    if feedback:
        model_prob = new_model.predict_proba(scaler.transform(x))
        if model_prob.shape[1] > 1:
            model_pred = model_prob[:, 1] > threshold
        else:
            model_pred = model_prob[:, 0] > threshold

        model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]
        model_fp_idx = np.where(np.logical_and(y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(y)
        model_pred[model_fp_idx] = 1
    else:
        model_prob = new_model.predict_proba(scaler.transform(x))
        model_pred = copy.deepcopy(y)
        model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]

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


def track_performance(new_model: Model, rate_tracker, x_eval: np.ndarray, y_eval: np.ndarray,
                      threshold: float, scaler: Transformer):
    if threshold is None:
        eval_prob = new_model.predict_proba(scaler.transform(x_eval))
        eval_pred = new_model.predict(scaler.transform(x_eval))
    else:
        eval_prob = new_model.predict_proba(scaler.transform(x_eval))

        if eval_prob.shape[1] > 1:
            eval_pred = eval_prob[:, 1] >= threshold
        else:
            eval_pred = eval_prob[:, 0] >= threshold

    rate_tracker.update_rates(y_eval, eval_pred, eval_prob)


def refit_scaler(scaler: Transformer, data_wrapper, update: bool):
    if update:
        x = data_wrapper.get_all_data_for_scaler_fit()

        scaler.fit(x)
