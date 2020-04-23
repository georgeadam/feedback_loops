from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from src.models.sklearn import lr, linear_svm, rbf_svm, adaboost, random_forest, xgboost, lr_online, evaluate
from src.models.pytorch import NN

from omegaconf import DictConfig, OmegaConf
from typing import Any, Callable
from src.utils.typing import Model


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
            return model_fn(num_features=num_features, **kwargs)

    return inside


def get_model_fn(args: DictConfig) -> Callable[[int], Model]:
    if args.type == "lr":
        return wrapped(lr, args.use_cv, args.cv, warm_start=args.warm_start, class_weight=args.class_weight)
    elif args.type == "lr_online":
        return wrapped(lr_online, args.use_cv, args.cv, warm_start=args.warm_start, class_weight=args.class_weight)
    elif args.type == "linear_svm":
        return wrapped(linear_svm, args.use_cv, args.cv, warm_start=args.warm_start, class_weight=args.class_weight)
    elif args.type == "nn":
        return wrapped(NN, args.use_cv, args.cv, lr=args.lr, online_lr=args.online_lr, iterations=args.iterations,
                       optimizer_name=args.optimizer, reset_optim=args.reset_optim, tol=args.tol,
                       hidden_layers=args.hidden_layers, activation=args.activation)
    elif args.type == "svm_rbf":
        return wrapped(rbf_svm, args.use_cv, args.cv)
    elif args.type == "rf":
        return wrapped(random_forest, args.use_cv, args.cv, class_weight=args.class_weight)
    elif args.type == "adaboost":
        return wrapped(adaboost, args.use_cv, args.cv)
    elif args.type == "xgboost":
        return wrapped(xgboost, args.use_cv, args.cv, warm_start=args.warm_start)
