import sklearn.ensemble as ensemble
from .utils import evaluate


def random_forest(num_features: int=2, class_weight: str=None):
    model = ensemble.RandomForestClassifier(class_weight=class_weight, n_jobs=4)
    model.evaluate = evaluate

    return model