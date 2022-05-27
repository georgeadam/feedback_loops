from sklearn.model_selection import GridSearchCV
from .sklearn import adaboost, xgboost, lr_regular, lr_sgd, knn, random_forest, svm_linear, svm_rbf, evaluate
from .pytorch import NN, NN_LFE, NN_LRE, QNet

from omegaconf import DictConfig, OmegaConf
from typing import Callable
from src.utils.typing import Model


TRADITIONAL_ML_MODEL_TYPES = ["svm_linear", "lr", "adaboost", "random_forest", "svm_rbf", "xgboost", "rf", "knn"]
NN_MODEL_TYPES = ["nn", "nn_lre", "nn_lfe", "nn_dro"]


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
        return wrapped(lr_regular, args.optim.use_cv, args.model.cv, warm_start=args.update_params.warm_start, class_weight=args.optim.class_weight)
    elif args.model.type == "lr_online":
        return wrapped(lr_sgd, args.optim.use_cv, args.model.cv, warm_start=args.update_params.warm_start, class_weight=args.optim.class_weight)
    elif args.model.type == "svm_linear":
        return wrapped(svm_linear, args.optim.use_cv, args.model.cv, warm_start=args.update_params.warm_start, class_weight=args.optim.class_weight)
    elif args.model.type == "nn" or args.model.type == "nn_dro":
        return wrapped(NN, False, None, hidden_layers=args.model.hidden_layers, activation=args.model.activation,
                       dropout=args.model.dropout, device=args.model.device)
    elif args.model.type == "nn_lre":
        return wrapped(NN_LRE, None, None, hidden_layers=args.model.hidden_layers, activation=args.model.activation,
                       device=args.model.device)
    elif args.model.type == "nn_lfe":
        return wrapped(NN_LFE, None, None, hidden_layers=args.model.hidden_layers, activation=args.model.activation,
                       device=args.model.device)
    elif args.model.type == "svm_rbf":
        return wrapped(svm_rbf, args.optim.use_cv, args.model.cv)
    elif args.model.type == "rf":
        return wrapped(random_forest, args.optim.use_cv, args.model.cv, class_weight=args.optim.class_weight)
    elif args.model.type == "adaboost":
        return wrapped(adaboost, args.optim.use_cv, args.model.cv)
    elif args.model.type == "xgboost":
        return wrapped(xgboost, args.optim.use_cv, args.model.cv, warm_start=args.update_params.warm_start)
    elif args.model.type == "q_net":
        return wrapped(QNet, False, None, hidden_layers=args.model.hidden_layers, activation=args.model.activation,
                       device=args.model.device)
    elif args.model.type == "knn":
        return wrapped(knn, args.optim.use_cv, args.model.cv, n_neighbors=args.model.n_neighbors)