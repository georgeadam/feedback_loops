from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from src.models.sklearn import lr, linear_svm, rbf_svm, adaboost, random_forest, xgboost, lr_online, evaluate
from src.models.pytorch import NNEWC
from src.models.nn import NN
from src.models.lre import NN_LRE

from omegaconf import DictConfig, OmegaConf
from typing import Any, Callable
from src.utils.typing import Model


TRADITIONAL_ML_MODEL_TYPES = ["linear_svm", "lr", "adaboost", "random_forest", "rbf_svm", "xgboost", "rf"]
NN_MODEL_TYPES = ["nn", "nn_lre"]

# def wrapped(fn: Callable[[Any], Model], **kwargs: Any) -> Callable:
#     def inside(num_features: int) -> Model:
#         return fn(num_features=num_features, **kwargs)
#
#     return inside


def wrapped(model_fn, use_cv, cv, **kwargs) -> Callable:
    def inside(num_features: int) -> Model:
        if use_cv:
            temp = model_fn(num_features=num_features, **kwargs)
            grid_search_temp = GridSearchCV(temp, OmegaConf.to_container(cv))

            grid_search_temp.evaluate = evaluate

            return grid_search_temp
        else:
            if "device" in kwargs.keys():
                return model_fn(num_features=num_features, **kwargs).to(kwargs["device"])
            else:
                return model_fn(num_features=num_features, **kwargs)

    return inside


def get_model_fn(args: DictConfig) -> Callable[[int], Model]:
    if args.model.type == "lr":
        return wrapped(lr, args.optim.use_cv, args.optim.cv, warm_start=args.update_params.warm_start, class_weight=args.optim.class_weight)
    elif args.model.type == "lr_online":
        return wrapped(lr_online, args.use_cv, args.cv, warm_start=args.warm_start, class_weight=args.class_weight)
    elif args.model.type == "linear_svm":
        return wrapped(linear_svm, args.use_cv, args.cv, warm_start=args.warm_start, class_weight=args.class_weight)
    elif args.model.type == "nn":
        return wrapped(NN, False, None, hidden_layers=args.model.hidden_layers, activation=args.model.activation,
                       device=args.model.device)
    elif args.model.type == "nn_lre":
        return wrapped(NN_LRE, None, None, hidden_layers=args.model.hidden_layers, activation=args.model.activation,
                       device=args.model.device)
    elif args.model.type == "svm_rbf":
        return wrapped(rbf_svm, args.optim.use_cv, args.optim.cv)
    elif args.model.type == "rf":
        return wrapped(random_forest, args.optim.use_cv, args.optim.cv, class_weight=args.optim.class_weight)
    elif args.model.type == "adaboost":
        return wrapped(adaboost, args.use_cv, args.cv)
    elif args.model.type == "xgboost":
        return wrapped(xgboost, args.use_cv, args.cv, warm_start=args.warm_start)
    elif args.model.type == "nn_ewc":
        return wrapped(NNEWC, args.use_cv, args.cv, lr=args.lr, online_lr=args.online_lr, iterations=args.iterations,
                       optimizer_name=args.optimizer, reset_optim=args.reset_optim, tol=args.tol,
                       hidden_layers=args.hidden_layers, activation=args.activation, importance=args.importance)
