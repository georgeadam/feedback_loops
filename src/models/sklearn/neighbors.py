from sklearn.neighbors import KNeighborsClassifier
from .utils import evaluate


def knn(num_features: int=2, class_weight: str=None, n_neighbors=1):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.evaluate = evaluate

    return model