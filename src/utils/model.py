from sklearn.calibration import CalibratedClassifierCV
from src.models.sklearn import lr, linear_svm, rbf_svm, adaboost, random_forest, xgboost, lr_online, evaluate
from src.models.pytorch import NN


def wrapped(fn, **kwargs):
    def inside(num_features):
        return fn(num_features=num_features, **kwargs)

    return inside


def get_model_fn(args):
    if args.model == "lr":
        return wrapped(lr, warm_start=args.warm_start, class_weight=args.class_weight)
    elif args.model == "lr_online":
        return wrapped(lr_online, warm_start=args.warm_start, class_weight=args.class_weight)
    elif args.model == "lr_pytorch":
        return wrapped(NN, iterations=args.iterations, lr=args.lr, online_lr=args.online_lr, optimizer_name=args.optimizer,
                       reset_optim=args.reset_optim, tol=args.tol, hidden_layers=args.hidden_layers,
                       activation=args.activation, soft=args.soft)
    elif args.model == "linear_svm":
        return wrapped(linear_svm, warm_start=args.warm_start, class_weight=args.class_weight)
    elif args.model == "nn":
        return wrapped(NN, lr=args.lr, online_lr=args.online_lr, iterations=args.iterations, optimizer_name=args.optimizer,
                       reset_optim=args.reset_optim, tol=args.tol, hidden_layers=args.hidden_layers, activation=args.activation)
    elif args.model == "svm_rbf":
        return rbf_svm
    elif args.model == "random_forest":
        return wrapped(random_forest, class_weight=args.class_weight)
    elif args.model == "adaboost":
        return adaboost
    elif args.model == "xgboost":
        return wrapped(xgboost, warm_start=args.warm_start)
