import numpy as np
import warnings

from src.utils.metrics import RateTracker, PredictionTracker
from src.utils.metrics import create_empty_rates, create_empty_predictions
from src.utils.preprocess import get_scaler
from src.utils.rand import set_seed
from src.scripts.experiment.threshold import find_threshold
from src.utils.typing import ResultDict, ModelFn, DataFn

from omegaconf import DictConfig
from typing import Dict, Callable, List, Optional


def get_dyanmic_desired_value(ddr: str, rates: Dict[str, List[float]]) -> Optional[float]:
    if ddr is not None and ddr == "f1":
        return 1.0
    elif ddr is not None and ddr == "f1_static":
        return rates["f1"][0]
    elif ddr is not None:
        return rates[ddr][0]

    return None


def experiment_loop(data_fn: DataFn=None, data_wrapper_fn=None, model_fn: ModelFn=None, trainer_fn=None,
                    update_fn: Callable=None, idr: str= "fpr", idv: float=0.1,
                    seeds: int=1, normalize: bool=True, return_model: bool=False, **kwargs) -> (ResultDict, List):
    seeds = np.arange(seeds)
    rates = create_empty_rates()
    predictions = create_empty_predictions()
    models = []

    for seed in seeds:
        print("Seed: {}".format(seed))
        set_seed(seed)

        # Create data, model, and trainer
        data, cols = data_fn()
        data_wrapper = data_wrapper_fn(data)
        model = model_fn(num_features=data_wrapper.dimension)
        trainer = trainer_fn(model_fn=model_fn, seed=seed)

        x_train, _ = data_wrapper.get_init_train_data()
        x_thresh, y_thresh = data_wrapper.get_init_thresh_data()
        x_eval, y_eval = data_wrapper.get_eval_data(0)

        scaler = get_scaler(normalize, cols)
        scaler.fit(x_train)

        # Fit model
        set_seed(seed)
        trainer.initial_fit(model, data_wrapper, scaler)
        y_prob = model.predict_proba(scaler.transform(x_thresh))

        if idr is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                model.threshold = find_threshold(y_thresh, y_prob, idr, idv)
        else:
            model.threshold = 0.5

        set_seed(seed)
        y_prob = model.predict_proba(scaler.transform(x_eval))
        y_pred = y_prob[:, 1] > model.threshold

        rate_tracker = RateTracker()
        rate_tracker.update_rates(y_eval, y_pred, y_prob, 0, "eval")

        prediction_tracker = PredictionTracker()
        prediction_tracker.update_predictions(x_eval, y_eval, y_pred.astype(int), y_prob[:, 1], 0, model.threshold, "eval")

        ddv = get_dyanmic_desired_value(data_wrapper.get_ddr(), rate_tracker.get_rates())

        # Update model
        model = update_fn(model, data_wrapper, rate_tracker, prediction_tracker, trainer=trainer,
                          ddv=ddv, scaler=scaler)
        temp_rates = rate_tracker.get_rates()
        temp_rates["seed"] = [seed] * len(temp_rates[list(temp_rates.keys())[0]])
        temp_predictions = prediction_tracker.get_predictions()
        temp_predictions["seed"] = [seed] * len(temp_predictions[list(temp_predictions.keys())[0]])

        if return_model:
            models.append(model.to("cpu"))

        for key in temp_rates.keys():
            rates[key] += temp_rates[key]

        for key in temp_predictions.keys():
            predictions[key] += temp_predictions[key]

    return rates, predictions, models


def call_experiment_loop(args: DictConfig, data_fn: Callable, data_wrapper_fn: Callable, model_fn: Callable,
                         trainer, update_fn: Callable) -> (ResultDict, ResultDict):
    return experiment_loop(data_fn, data_wrapper_fn, model_fn, trainer, update_fn,
                           idr=args.rates.idr, idv=args.rates.idv, seeds=args.misc.seeds,
                           normalize=args.data.normalize, return_model=args.misc.return_model)
