import copy as copylib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.typing import Transformer
from typing import Any, List


class NullScaler(object):
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def fit(self, x: np.ndarray):
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class CustomStandardScaler(StandardScaler):
    def __init__(self, cols: List[int]=None):
        super(CustomStandardScaler, self).__init__()
        self.cols = cols

    def fit(self, X: np.ndarray, y: np.ndarray=None) -> None:
        if self.cols is not None:
            super(CustomStandardScaler, self).fit(X[:, self.cols])
        else:
            super(CustomStandardScaler, self).fit(X)

    def transform(self, X: np.ndarray, copy=None) -> np.ndarray:
        if self.cols is not None:
            X_copy = copylib.deepcopy(X)
            orig_shape = X_copy.shape

            if len(orig_shape) < 2:
                X_copy = X_copy.reshape(1, -1)

            X_copy[:, self.cols] = super(CustomStandardScaler, self).transform(X_copy[:, self.cols], copy=copy)

            X_copy = X_copy.reshape(*orig_shape)

            return X_copy
        else:
            orig_shape = X.shape

            if len(orig_shape) < 2:
                X = X.reshape(1, -1)

            X = X.reshape(*orig_shape)

            return super(CustomStandardScaler, self).transform(X)


def get_scaler(normalize: bool, cols: List[int]=None) -> Transformer:
    if normalize:
        return CustomStandardScaler(cols)
    else:
        return NullScaler()
