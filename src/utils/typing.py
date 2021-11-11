from abc import ABC, abstractmethod
import numpy as np

from typing import Any, Callable, Dict, Union


class Model(ABC):
    threshold = 0.5
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        pass


class Transformer(ABC):
    def fit(self, x: np.ndarray, y=None):
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        pass


IntOrFloat = Union[int, float]
ResultDict = Dict[str, Dict[str, float]]
DataFn = Callable[[IntOrFloat, IntOrFloat, IntOrFloat, Any], Any]
ModelFn = Callable[[Any], Model]