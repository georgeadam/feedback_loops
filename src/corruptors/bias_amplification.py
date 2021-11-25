import copy
import numpy as np


class BiasAmplification(object):
    def __init__(self):
        pass

    def __call__(self, model, x, y, scaler):
        model_prob = model.predict_proba(scaler.transform(x))

        if model_prob.shape[1] > 1:
            model_pred = model_prob[:, 1] > model.threshold
        else:
            model_pred = model_prob[:, 0] > model.threshold

        model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]
        model_fp_idx = np.where(np.logical_and(y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(y)
        model_pred[model_fp_idx] = 1

        return model_pred, model_prob