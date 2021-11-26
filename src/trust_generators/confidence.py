import numpy as np


class Confidence(object):
    def __init__(self):
        pass

    def __call__(self, model_prob: np.ndarray, *args, **kwargs) -> np.ndarray:
        return model_prob