import numpy as np
from sklearn.base import BaseEstimator

import torch.nn


class NN(torch.nn.Module, BaseEstimator):
    def __init__(self, num_features, hidden_layers, activation, dropout=0.0, device="cpu"):
        super(NN, self).__init__()

        self.activation = getattr(torch.nn, activation)()
        self.num_features = num_features
        self.layers = self._create_layers(hidden_layers)
        self.dropout = dropout
        self.device = device
        self.classes_ = np.array([0, 1])
        self._threshold = 0.5

    def forward(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

            if self.dropout > 0:
                x = torch.nn.Dropout(self.dropout)(x)


        x = self.layers[-1](x)

        return x

    def predict(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            out = self.forward(x)

        pred = torch.max(out, 1)[1]

        return pred.detach().cpu().numpy()

    def predict_proba(self, X):
        if type(X) is np.ndarray:
            X = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            out = self.forward(X)

        return torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()

    def predict_proba_grad(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)

        out = self.forward(x)

        return torch.nn.functional.softmax(out, dim=1)

    def _create_layers(self, hidden_layers):
        if hidden_layers == 0:
            fc = torch.nn.ModuleList([torch.nn.Linear(self.num_features, 2)])
        elif hidden_layers == 1:
            fc = torch.nn.ModuleList([torch.nn.Linear(self.num_features, 10),
                                      torch.nn.Linear(10, 2)])
        elif hidden_layers == 2:
            fc = torch.nn.ModuleList([torch.nn.Linear(self.num_features, 20),
                                      torch.nn.Linear(20, 10),
                                      torch.nn.Linear(10, 2)])
        elif hidden_layers > 2:
            initial_hidden_units = 50 * (2 ** hidden_layers)
            fc = [torch.nn.Linear(self.num_features, initial_hidden_units)]
            prev_hidden_units = initial_hidden_units

            for i in range(1, hidden_layers):
                next_hidden_units = int(initial_hidden_units / (2 ** i))
                fc.append(torch.nn.Linear(prev_hidden_units, next_hidden_units))

                prev_hidden_units = next_hidden_units

            fc.append(torch.nn.Linear(prev_hidden_units, 2))
            fc = torch.nn.ModuleList(fc)

        return fc

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        pass

    def fit(self, *args):
        pass

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold):
        self._threshold = new_threshold
