import copy
import numpy as np

from src.scripts.experiment.threshold import refit_threshold
from src.utils.metrics import compute_model_fpr

from src.corruptors import get_corruptor
from src.experts import get_expert
from src.trust_generators import get_trust_generator
from src.utils.typing import Model, Transformer
from typing import Callable, Optional


def wrapped(fn, **kwargs):
    def inside(*args, **specified_args):
        return fn(*args, **kwargs, **specified_args)

    return inside


def get_update_fn(args):
    update_fn = update_model_general
    corruptor = get_corruptor(args)
    expert = get_expert(args)
    trust_generator = get_trust_generator(args)

    return wrapped(update_fn, corruptor=corruptor, expert=expert, trust_generator=trust_generator,
                   update=args.update_params.do_update)


def update_model_general(model, data_wrapper, rate_tracker, prediction_tracker, trainer=None, corruptor=None, expert=None, trust_generator=None,
                         update: bool = True,  ddv: Optional[float] = None, scaler: Transformer = None):
    x_train, y_train = data_wrapper.get_train_data()

    for update_num, (x_update, y_update) in enumerate(data_wrapper.get_update_data_generator(), start=1):
        # Create copies of experiment labels so we don't overwrite original data
        y_update = copy.deepcopy(y_update)
        y_update_unmodified = copy.deepcopy(y_update)

        # Compute FPR on experiment data to be used by conditional clinician trust
        y_update = replace_labels(model, x_update, y_update, scaler, corruptor, expert, trust_generator)

        data_wrapper.store_current_update_batch_corrupt(x_update, y_update)
        data_wrapper.store_current_update_batch_clean(x_update, y_update_unmodified)

        # Update data normalization to take into account new data
        refit_scaler(scaler, data_wrapper, update)
        model = trainer.update_fit(model, data_wrapper, rate_tracker, scaler, update_num)
        model.threshold = refit_threshold(model, data_wrapper, ddv, scaler, update)

        x_eval, y_eval = data_wrapper.get_eval_data(update_num)
        track_performance(model, rate_tracker, prediction_tracker, x_eval, y_eval, scaler, update_num)

        data_wrapper.accumulate_update_data()
        rate_tracker.update_detection(y_train, data_wrapper.get_cumulative_update_data()[1])

    return model


def replace_labels(model: Model, x: np.ndarray, y: np.ndarray, scaler: Transformer,
                   corruptor, expert, trust_generator):
    model_fpr = compute_model_fpr(model, x, y, scaler)
    model_pred, model_prob = corruptor(model, x, y, scaler)
    trust = trust_generator(model_fpr=model_fpr, model_prob=model_prob)
    expert_pred = expert(y)

    if type(trust) != float:
        bernoulli = np.random.binomial([1] * len(y), trust)
    else:
        bernoulli = np.random.choice([0, 1], len(y), p=[1 - trust, trust])

    new_labels = bernoulli * model_pred + (1 - bernoulli) * expert_pred

    return new_labels


def compute_model_pred(new_model, scaler, x, y):
    model_prob = new_model.predict_proba(scaler.transform(x))
    model_pred = copy.deepcopy(y)
    model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]

    return model_pred, model_prob


# def track_performance(model: Model, rate_tracker, prediction_tracker,
#                       x_eval: np.ndarray, y_eval: np.ndarray, scaler: Transformer, update_num):
#     if model.threshold is None:
#         eval_prob = model.predict_proba(scaler.transform(x_eval))
#         eval_pred = model.predict(scaler.transform(x_eval))
#     else:
#         eval_prob = model.predict_proba(scaler.transform(x_eval))
#
#         if eval_prob.shape[1] > 1:
#             eval_pred = eval_prob[:, 1] >= model.threshold
#         else:
#             eval_pred = eval_prob[:, 0] >= model.threshold
#
#     rate_tracker.update_rates(y_eval, eval_pred, eval_prob)
#     prediction_tracker.update_predictions(y_eval, eval_pred.astype(int), eval_prob[:, 1], update_num, model.threshold)


def track_performance(model: Model, rate_tracker, prediction_tracker,
                      x_eval: np.ndarray, y_eval: np.ndarray, scaler: Transformer, update_num):
    if model.threshold is None:
        eval_prob = model.predict_proba(scaler.transform(x_eval))
        eval_pred = model.predict(scaler.transform(x_eval))
    else:
        eval_prob = model.predict_proba(scaler.transform(x_eval))

        if eval_prob.shape[1] > 1:
            eval_pred = eval_prob[:, 1] >= model.threshold
        else:
            eval_pred = eval_prob[:, 0] >= model.threshold

    rate_tracker.update_rates(y_eval, eval_pred, eval_prob)
    prediction_tracker.update_predictions(x_eval, y_eval, eval_pred.astype(int), eval_prob[:, 1], update_num, model.threshold)


def refit_scaler(scaler: Transformer, data_wrapper, update: bool):
    if update:
        x = data_wrapper.get_all_data_for_scaler_fit()

        scaler.fit(x)
