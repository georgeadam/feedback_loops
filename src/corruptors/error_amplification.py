import copy

import numpy as np


class ErrorAmplification(object):
    def __init__(self, fn_corruption_prob, fp_corruption_prob):
        self._validate_corruption_prob(fn_corruption_prob)
        self._validate_corruption_prob(fp_corruption_prob)

        self.fn_corruption_prob = fn_corruption_prob
        self.fp_corruption_prob = fp_corruption_prob

    def __call__(self, model, x, y, scaler):
        y = copy.deepcopy(y)
        model_prob = model.predict_proba(scaler.transform(x))

        if model_prob.shape[1] > 1:
            model_pred = model_prob[:, 1] > model.threshold
        else:
            model_pred = model_prob[:, 0] > model.threshold

        model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]

        if self.fn_corruption_prob > 0:
            model_fn_idx = np.where(np.logical_and(y == 1, model_pred == 0))[0]
            model_fn_idx = np.random.choice(model_fn_idx, size=int(self.fn_corruption_prob * len(model_fn_idx)), replace=False)
            y[model_fn_idx] = 0

        if self.fp_corruption_prob > 0:
            model_fp_idx = np.where(np.logical_and(y == 0, model_pred == 1))[0]
            model_fp_idx = np.random.choice(model_fp_idx, size=int(self.fp_corruption_prob * len(model_fp_idx)), replace=False)
            y[model_fp_idx] = 1

        return y, model_prob

    def _validate_corruption_prob(self, corruption_prob):
        assert isinstance(corruption_prob, float), "Corruption prob should be a number but got {}".format(type(corruption_prob))
        assert corruption_prob <= 1.0, "Corruption prob should be less than 1.0 but got {}".format(corruption_prob)
        assert corruption_prob >= 0.0, "Corruption prob should be greater than 0.0 but got {}".format(corruption_prob)