from sklearn.calibration import CalibratedClassifierCV
from src.models.sklearn import lr, linear_svm, rbf_svm, adaboost, random_forest, xgboost, lr_online, evaluate
from src.models.pytorch import NN


def wrapped(fn, **kwargs):
    def inside(num_features):
        return fn(num_features=num_features, **kwargs)

    return inside


def get_model_fn(model_args, pytorch_args):
    if model_args.type == "lr":
        return wrapped(lr, warm_start=model_args.warm_start, class_weight=model_args.class_weight)
    elif model_args.type == "lr_online":
        return wrapped(lr_online, warm_start=model_args.warm_start, class_weight=model_args.class_weight)
    elif model_args.type == "lr_pytorch":
        return wrapped(NN, iterations=pytorch_args.iterations, lr=pytorch_args.lr, online_lr=pytorch_args.online_lr,
                       optimizer_name=pytorch_args.optimizer, reset_optim=pytorch_args.reset_optim, tol=pytorch_args.tol,
                       hidden_layers=pytorch_args.hidden_layers, activation=pytorch_args.activation, soft=pytorch_args.soft)
    elif model_args.type == "linear_svm":
        return wrapped(linear_svm, warm_start=model_args.warm_start, class_weight=model_args.class_weight)
    elif model_args.type == "nn":
        return wrapped(NN, lr=pytorch_args.lr, online_lr=pytorch_args.online_lr, iterations=pytorch_args.iterations,
                       optimizer_name=pytorch_args.optimizer, reset_optim=pytorch_args.reset_optim, tol=pytorch_args.tol,
                       hidden_layers=pytorch_args.hidden_layers, activation=pytorch_args.activation)
    elif model_args.type == "svm_rbf":
        return rbf_svm
    elif model_args.type == "random_forest":
        return wrapped(random_forest, class_weight=model_args.class_weight)
    elif model_args.type == "adaboost":
        return adaboost
    elif model_args.type == "xgboost":
        return wrapped(xgboost, warm_start=model_args.warm_start)
