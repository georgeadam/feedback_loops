from src.models.sklearn import lr, linear_svm
from src.models.pytorch import LR, LREWC, NN


def wrapped(fn, **kwargs):
    def inside(num_features):
        return fn(num_features=num_features, **kwargs)

    return inside


def get_model_fn(args):
    if args.model == "lr":
        return lr
    elif args.model == "linear_svm":
        return linear_svm
    elif args.model == "lr_pytorch":
        return wrapped(LR, lr=args.lr, iterations=args.iterations)
    elif args.model == "lr_ewc":
        return wrapped(LREWC, lr=args.lr, iterations=args.iterations, importance=args.importance)
    elif args.model == "nn":
        return wrapped(NN, lr=args.lr, iterations=args.iterations)