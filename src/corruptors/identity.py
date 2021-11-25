import copy
import numpy as np


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, model, x, y, scaler):
        model_pred = copy.deepcopy(y)
        model_prob = model.predict_proba(scaler.transform(x))
        model_prob = model_prob[np.arange(len(model_prob)), model_pred.astype(int)]

        return model_pred, model_prob