import copy
import numpy as np


class UniformNoise(object):
    def __init__(self, noise):
        self.noise = noise

    def __call__(self, model, x, y, scaler):
        y = copy.deepcopy(y)
        flip_indices = np.random.choice(len(y), int(self.noise * len(y)), replace=False)
        y[flip_indices] = 1 - y[flip_indices]

        return y, y