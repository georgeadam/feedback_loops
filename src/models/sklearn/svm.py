import sklearn.svm as svm
from .utils import evaluate


def svm_linear(num_features: int=2, class_weight: str=None, warm_start: bool=True):
    model = svm.SVC(probability=True, kernel="linear")
    model.evaluate = evaluate
    model.threshold = 0.5

    return model


def svm_rbf(num_features: int=2, class_weight: str=None):
    model = svm.SVC(probability=True, gamma="auto")
    model.evaluate = evaluate
    model.threshold = 0.5

    return model