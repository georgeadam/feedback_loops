import copy
import numpy as np


class RandomError(object):
    def __init__(self, fnr: float, fpr: float):
        self._validate_error_rate(fnr)
        self._validate_error_rate(fpr)

        self.fnr = fnr
        self.fpr = fpr

    def __call__(self, y):
        expert_pred = copy.deepcopy(y)

        neg_idx = np.where(expert_pred == 0)[0]
        pos_idx = np.where(expert_pred == 1)[0]

        if self.fpr > 0:
            expert_fp_idx = np.random.choice(neg_idx, min(int(self.fpr * len(neg_idx)), len(neg_idx)))
            expert_pred[expert_fp_idx] = 1

        if self.fnr > 0:
            expert_fn_idx = np.random.choice(pos_idx, min(int(self.fnr * len(pos_idx)), len(pos_idx)))
            expert_pred[expert_fn_idx] = 0

        return expert_pred

    def _validate_error_rate(self, error_rate):
        assert isinstance(error_rate, float), "Error rate should be a number but got {}".format(type(error_rate))