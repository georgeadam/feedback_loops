import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = grad.detach().data

                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = grad.detach().data

                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            if type(getattr(curr_mod, name)) == nn.parameter.Parameter and not (type(param) == nn.parameter.Parameter):
                setattr(curr_mod, name, nn.Parameter(param))
            else:
                setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = param.data.clone()
                param.requires_grad = True
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, device, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        # self.weight = nn.Parameter(ignore.weight)
        # self.bias = nn.Parameter(ignore.bias)
        self.register_buffer('weight', create_buffer(ignore.weight))
        self.register_buffer('bias', create_buffer(ignore.bias))

        self.weight = self.weight.to(device).detach().requires_grad_(True)
        self.bias = self.bias.to(device).detach().requires_grad_(True)
        # self.weight = ignore.weight.requires_grad_(True)
        # self.bias = ignore.bias.requires_grad_(True)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def create_buffer(p):
    buffer = p.data
    buffer.requires_grad = True

    return buffer


class NN_LRE(MetaModule):
    def __init__(self, num_features, hidden_layers, activation, device):
        super(NN_LRE, self).__init__()

        self.activation = getattr(nn, activation)()
        self.num_features = num_features
        self.layers = self._create_layers(hidden_layers, device)
        self.device = device

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
