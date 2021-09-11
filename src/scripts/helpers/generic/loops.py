import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.metrics import compute_all_rates, RateTracker
from src.utils.misc import create_empty_rates
from src.utils.preprocess import get_scaler
from src.utils.rand import set_seed
from src.utils.threshold import find_threshold
from src.utils.typing import ResultDict, ModelFn, DataFn

from omegaconf import DictConfig
from typing import Dict, Callable, List, Optional


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


def get_dyanmic_desired_value(ddr: str, rates: Dict[str, List[float]]) -> Optional[float]:
    if ddr is not None and ddr == "f1":
        return 1.0
    elif ddr is not None:
        return rates[ddr][0]

    return None


def train_update_loop(data_fn: DataFn=None, data_wrapper_fn=None, model_fn: ModelFn=None, trainer_fn=None,
                      update_fn: Callable=None, idr: str= "fpr", idv: float=0.1,
                      seeds: int=1, clinician_fpr: float=0.0, clinician_trust: float=1.0,
                      normalize: bool=True, recover_prob: float=1.0, return_model: bool=False,  **kwargs) -> (ResultDict, List):
    seeds = np.arange(seeds)
    rates = create_empty_rates()
    models = []

    for seed in seeds:
        print("Seed: {}".format(seed))
        set_seed(seed)

        data, cols = data_fn()
        data_wrapper = data_wrapper_fn(data)
        model = model_fn(num_features=data_wrapper.dimension)
        trainer = trainer_fn(model_fn=model_fn, seed=seed)

        x_train, _ = data_wrapper.get_init_train_data()
        x_thresh, y_thresh = data_wrapper.get_init_thresh_data()
        x_eval, y_eval = data_wrapper.get_eval_data(0)

        scaler = get_scaler(normalize, cols)
        scaler.fit(x_train)

        set_seed(seed)
        trainer.initial_fit(model, data_wrapper, scaler)
        y_prob = model.predict_proba(scaler.transform(x_thresh))

        if idr is not None:
            threshold = find_threshold(y_thresh, y_prob, idr, idv)
        else:
            threshold = 0.5

        set_seed(seed)
        y_prob = model.predict_proba(scaler.transform(x_eval))
        y_pred = y_prob[:, 1] > threshold

        rate_tracker = RateTracker()
        rate_tracker.update_rates(y_eval, y_pred, y_prob)

        ddv = get_dyanmic_desired_value(data_wrapper.get_ddr(), rate_tracker.get_rates())

        model = update_fn(model, data_wrapper, rate_tracker, trainer=trainer, threshold=threshold,
                  ddv=ddv, clinician_fpr=clinician_fpr, clinician_trust=clinician_trust,
                  scaler=scaler, recover_prob=recover_prob)
        temp_rates = rate_tracker.get_rates()

        if return_model:
            models.append(model.to("cpu"))

        for key in temp_rates.keys():
            rates[key].append(temp_rates[key])

    return rates, models


def call_update_loop(args: DictConfig, data_fn: Callable, data_wrapper_fn: Callable, model_fn: Callable,
                     trainer, update_fn: Callable) -> (ResultDict, ResultDict):
    return train_update_loop(data_fn, data_wrapper_fn, model_fn, trainer, update_fn,
                             idr=args.rates.idr, idv=args.rates.idv, seeds=args.misc.seeds,
                             clinician_fpr=args.rates.clinician_fpr,
                             clinician_trust=args.trust.clinician_trust, normalize=args.data.normalize,
                             recover_prob=args.rates.recover_prob, return_model=args.misc.return_model)
