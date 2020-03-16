from src.models.sklearn import lr, linear_svm, rbf_svm, adaboost, random_forest, xgboost
from src.models.pytorch import NN


def wrapped(fn, **kwargs):
    def inside(num_features):
        return fn(num_features=num_features, **kwargs)

    return inside


def get_model_fn(args):
    if args.model == "lr":
        return lr
    elif args.model == "linear_svm":
        return linear_svm
    elif args.model == "nn":
        return wrapped(NN, lr=args.lr, online_lr=args.online_lr, iterations=args.iterations, optimizer_name=args.optimizer,
                       reset_optim=args.reset_optim, tol=args.tol, hidden_layers=args.hidden_layers, activation=args.activation)
    elif args.model == "svm_rbf":
        return rbf_svm
    elif args.model == "random_forest":
        return random_forest
    elif args.model == "adaboost":
        return adaboost
    elif args.model == "xgboost":
        return xgboost


def get_model_fn_specific_args(model, **kwargs):
    if args.model == "lr":
        return