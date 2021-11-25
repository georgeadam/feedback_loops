import copy
import numpy as np


class BiasOscillation(object):
    def __init__(self, corruption_prob):
        self.corruption_prob = corruption_prob

    def __call__(self, model, x, y, scaler):
        model_prob = model.predict_proba(scaler.transform(x))

        if model_prob.shape[1] > 1:
            model_pred = model_prob[:, 1] > model.threshold
        else:
            model_pred = model_prob[:, 0] > model.threshold

        model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]
        model_tp_idx = np.where(np.logical_and(y == 1, model_pred == 1))[0]
        idx = np.random.choice(model_tp_idx, int(self.corruption_prob * len(model_tp_idx)))
        model_pred = copy.deepcopy(y)
        model_pred[idx] = 0

        return model_pred, model_prob