import copy
import numpy as np


class BiasedNoise(object):
    def __init__(self, noise, direction):
        self.noise = noise
        self.direction = direction

    def __call__(self, model, x, y, scaler):
        y = copy.deepcopy(y)

        if self.direction == "positive":
            flip_indices = np.random.choice(np.where(y == 0)[0], int(self.noise * len(y)), replace=False)
            y[flip_indices] = 1 - y[flip_indices]
        elif self.direction == "negative":
            flip_indices = np.random.choice(np.where(y == 1)[0], int(self.noise * len(y)), replace=False)
            y[flip_indices] = 1 - y[flip_indices]
        else:
            raise Exception("Specified bias direction must be one of: [positive, negative], {} was provided".format(self.direction))

        return y, y