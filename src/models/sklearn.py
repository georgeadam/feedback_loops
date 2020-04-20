import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.naive_bayes as naive_bayes
import sklearn.svm as svm

import xgboost as xgb

from src.utils.typing import Model

def lr(num_features: int=2, class_weight: str=None, warm_start: bool=False) -> Model:
    model = linear_model.LogisticRegression(max_iter=10000, tol=1e-3, warm_start=warm_start, class_weight=class_weight,
                                            penalty="none")
    model.evaluate = evaluate

    return model


def lr_online(num_features: int=2, class_weight: str=None, warm_start: bool=False) -> Model:
    model = linear_model.SGDClassifier(loss="log", max_iter=10000, tol=1e-3, warm_start=warm_start,
                                       class_weight=class_weight)

    model.evaluate = evaluate

    return model


def linear_svm(num_features: int=2, class_weight: str=None, warm_start: bool=True) -> Model:
    model = linear_model.SGDClassifier(max_iter=10000, tol=1e-3, warm_start=warm_start, loss="hinge",
                                       class_weight=class_weight)
    model.evaluate = evaluate

    return model


def rbf_svm(num_features: int=2, class_weight: str=None) -> Model:
    model = svm.SVC(probability=True, gamma="auto")
    model.evaluate = evaluate

    return model


def random_forest(num_features: int=2, class_weight: str=None) -> Model:
    model = ensemble.RandomForestClassifier(class_weight=class_weight, n_jobs=4)
    model.evaluate = evaluate

    return model


def adaboost(num_features: int=2, class_weight: str=None) -> Model:
    model = ensemble.AdaBoostClassifier()
    model.evaluate = evaluate

    return model


def xgboost(num_features: int=2, class_weight: str=None, warm_start: bool=True) -> Model:
    # model = ensemble.GradientBoostingClassifier(warm_start=warm_start)
    model = xgb.sklearn.XGBClassifier(n_jobs=6, learning_rate=0.1, max_depth=3)
    model.evaluate = evaluate

    return model


def evaluate(x, y):
    return 0.0