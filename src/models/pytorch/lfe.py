import numpy as np
import torch
from .meta import MetaLinear, MetaModule
from torch import nn as nn


class NN_LFE(MetaModule):
    def __init__(self, num_features, hidden_layers, activation, device):
        super(NN_LFE, self).__init__()

        self.activation = getattr(nn, activation)()
        self.num_features = num_features
        self.layers = self._create_layers(hidden_layers, device)
        self.device = device
        self._threshold = 0.5

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        x = self.layers[-1](x)

        return x

    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            out = self.forward(x)

        return torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()

    def _create_layers(self, hidden_layers, device):
        if hidden_layers == 0:
            fc = nn.ModuleList([MetaLinear(device, self.num_features, 2)])
        elif hidden_layers == 1:
            fc = nn.ModuleList([MetaLinear(device, self.num_features, 10),
                                MetaLinear(device, 10, 2)])
        elif hidden_layers == 2:
            fc = nn.ModuleList([MetaLinear(device, self.num_features, 20),
                                MetaLinear(device, 20, 10),
                                MetaLinear(device, 10, 2)])
        elif hidden_layers > 2:
            initial_hidden_units = 50 * (2 ** hidden_layers)
            fc = [MetaLinear(device, self.num_features, initial_hidden_units)]
            prev_hidden_units = initial_hidden_units

            for i in range(1, hidden_layers):
                next_hidden_units = int(initial_hidden_units / (2 ** i))
                fc.append(MetaLinear(device, prev_hidden_units, next_hidden_units))

                prev_hidden_units = next_hidden_units

            fc.append(MetaLinear(device, prev_hidden_units, 2))
            fc = nn.ModuleList(fc)

        return fc

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold):
        self._threshold = new_threshold
