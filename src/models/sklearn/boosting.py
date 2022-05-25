import sklearn.ensemble as ensemble
# import xgboost as xgb
from .utils import evaluate


def adaboost(num_features: int=2, class_weight: str=None):
    model = ensemble.AdaBoostClassifier()
    model.evaluate = evaluate
    model.threshold = 0.5

    return model


def xgboost(num_features: int=2, class_weight: str=None, warm_start: bool=True):
    model = ensemble.GradientBoostingClassifier(warm_start=warm_start)
    # model = xgb.sklearn.XGBClassifier(n_jobs=6, learning_rate=0.1, max_depth=3)
    model.evaluate = evaluate
    model.threshold = 0.5

    return model