import sklearn.linear_model as linear_model
from .utils import evaluate


def lr_regular(num_features: int=2, class_weight: str=None, warm_start: bool=False):
    model = linear_model.LogisticRegression(max_iter=10000, tol=1e-3, warm_start=warm_start, class_weight=class_weight,
                                            penalty="none")
    model.evaluate = evaluate

    return model


def lr_sgd(num_features: int=2, class_weight: str=None, warm_start: bool=False):
    model = linear_model.SGDClassifier(loss="log", max_iter=10000, tol=1e-3, warm_start=warm_start,
                                       class_weight=class_weight)

    model.evaluate = evaluate

    return model