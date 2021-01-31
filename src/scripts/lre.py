import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


import hydra.experimental
import os
import sys
sys.path.append("..")

from omegaconf import DictConfig
from settings import ROOT_DIR

from sklearn.model_selection import train_test_split

from src.utils.metrics import compute_all_rates
from src.utils.preprocess import get_scaler
from src.utils.rand import set_seed
from src.utils.data import get_data_fn
from src.utils.save import CSV_FILE


os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs/lre.yaml")


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
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', create_buffer(ignore.weight))
        self.register_buffer('bias', create_buffer(ignore.bias))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


def create_buffer(p):
    buffer = p.data
    buffer.requires_grad = True

    return buffer


class NN_Meta(MetaModule):
    def __init__(self, num_features, num_classes, num_layers, activation):
        super(NN_Meta, self).__init__()

        self.activation = getattr(nn, activation)()
        self.num_features = num_features
        self.layers = self._create_layers(num_layers, num_classes)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        x = self.layers[-1](x)

        return x

    def _create_layers(self, hidden_layers, num_classes):
        if hidden_layers == 0:
            fc = nn.ModuleList([MetaLinear(self.num_features, num_classes)])
        elif hidden_layers == 1:
            fc = nn.ModuleList([MetaLinear(self.num_features, 10),
                                MetaLinear(10, num_classes)])
        elif hidden_layers == 2:
            fc = nn.ModuleList([MetaLinear(self.num_features, 20),
                                MetaLinear(20, 10),
                                MetaLinear(10, num_classes)])
        elif hidden_layers > 2:
            initial_hidden_units = 50 * (2 ** hidden_layers)
            fc = [MetaLinear(self.num_features, initial_hidden_units)]
            prev_hidden_units = initial_hidden_units

            for i in range(1, hidden_layers):
                next_hidden_units = int(initial_hidden_units / (2 ** i))
                fc.append(MetaLinear(prev_hidden_units, next_hidden_units))

                prev_hidden_units = next_hidden_units

            fc.append(MetaLinear(prev_hidden_units, num_classes))
            fc = nn.ModuleList(fc)

        return fc


def load_data(data, model):
    data_fn = get_data_fn(data, model)
    x_train, y_train, x_update, y_update, x_test, y_test, cols = data_fn(20, 140, 1000, num_features=data.num_features)
    x_update, x_val, y_update, y_val = train_test_split(x_update, y_update, test_size=0.2)

    scaler = get_scaler(True, cols)
    scaler.fit(x_train)

    x_train_torch = torch.from_numpy(scaler.transform(x_train)).float().to(model.device)
    x_val_torch = torch.from_numpy(scaler.transform(x_val)).float().to(model.device)
    x_update_torch = torch.from_numpy(scaler.transform(x_update)).float().to(model.device)
    x_test_torch = torch.from_numpy(scaler.transform(x_test)).float().to(model.device)

    y_train_torch = torch.from_numpy(y_train).long().to(model.device)
    y_val_torch = torch.from_numpy(y_val).long().to(model.device)
    y_update_torch = torch.from_numpy(y_update).long().to(model.device)
    y_test_torch = torch.from_numpy(y_test).long().to(model.device)

    return (x_train_torch, y_train_torch, x_val_torch, y_val_torch,
            x_update_torch, y_update_torch, x_test_torch, y_test_torch)


def train_regular(model, x, y, optimizer, epochs):
    model.train()
    losses = []

    for i in range(epochs):
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


def train_lre(model, x_train, y_train, x_val, y_val, optimizer, epochs):
    meta_losses_clean = []
    net_losses = []

    smoothing_alpha = 0.9
    meta_l = 0
    net_l = 0

    for i in range(epochs):
        model.train()
        # Line 2 get batch of data
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_model = NN_Meta(x_train.shape[1], 2, 4, "ReLU")
        meta_model.load_state_dict(model.state_dict())
        meta_model = meta_model.to(x_train.device)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_model(x_train)
        cost = F.cross_entropy(y_f_hat, y_train, reduce=False)
        eps = torch.zeros(cost.size()).to(x_train.device)
        eps.requires_grad = True

        l_f_meta = torch.sum(cost * eps)

        meta_model.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_model.update_params(optimizer.param_groups[0]["lr"], source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_model(x_val)

        l_g_meta = F.cross_entropy(y_g_hat, y_val)

        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = model(x_train)
        cost = F.cross_entropy(y_f_hat, y_train, reduce=False)
        l_f = torch.sum(cost * w)

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
        meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (i + 1)))

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
        net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

    return meta_losses_clean, net_losses, w


def eval_model(model, x, y):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    y_out = model(x)
    y_prob = softmax(y_out)
    y_pred = torch.max(y_out, 1)[1]

    rates = compute_all_rates(y.detach().numpy(), y_pred.detach().numpy(),
                                        y_prob.detach().numpy())
    return rates


def create_corrupted_labels(model, x, y):
    y_out = model(x)
    y_pred = torch.max(y_out, 1)[1]

    fps = np.where((y == 0) & (y_pred == 1))

    y_corrupt = copy.deepcopy(y)
    y_corrupt[fps] = 1
    y_corrupt_torch = y_corrupt.long().to(y.device)

    return y_corrupt_torch


@hydra.main(config_path=config_path)
def main(args: DictConfig):
    print(args.pretty())
    print("Saving to: {}".format(os.getcwd()))
    results = {"train_type": []}

    for seed in range(args.misc.seeds):
        print("Running seed: {}".format(seed))
        set_seed(seed)
        (x_train_torch, y_train_torch, x_val_torch, y_val_torch,
         x_update_torch, y_update_torch, x_test_torch, y_test_torch) = load_data(args.data, args.model)

        set_seed(seed)
        regular_model = NN_Meta(x_train_torch.shape[1], 2, args.model.hidden_layers,
                                args.model.activation).to(args.model.device)

        optimizer = torch.optim.SGD(regular_model.params(), args.optim.lr, momentum=args.optim.momentum,
                                    weight_decay=args.optim.weight_decay)
        _ = train_regular(regular_model, x_train_torch, y_train_torch, optimizer, args.optim.epochs)
        regular_rates = eval_model(regular_model, x_test_torch, y_test_torch)

        y_update_corrupt_torch = create_corrupted_labels(regular_model, x_update_torch, y_update_torch)

        x_fused_torch = torch.cat([x_train_torch, x_update_torch]).to(args.model.device)
        y_fused_clean_torch = torch.cat([y_train_torch, y_update_torch]).to(args.model.device)
        y_fused_corrupt_torch = torch.cat([y_train_torch, y_update_corrupt_torch]).to(args.model.device)

        set_seed(seed)
        regular_clean_model = NN_Meta(x_fused_torch.shape[1], 2, 4, "ReLU")
        optimizer = torch.optim.SGD(regular_clean_model.params(), args.optim.lr)
        _ = train_regular(regular_clean_model, x_fused_torch, y_fused_clean_torch, optimizer, args.optim.epochs)
        regular_clean_rates = eval_model(regular_clean_model, x_test_torch, y_test_torch)

        set_seed(seed)
        regular_corrupt_model = NN_Meta(x_fused_torch.shape[1], 2, 4, "ReLU")
        optimizer = torch.optim.SGD(regular_corrupt_model.params(), args.optim.lr)
        _ = train_regular(regular_corrupt_model, x_fused_torch, y_fused_corrupt_torch, optimizer, args.optim.epochs)
        regular_corrupt_rates = eval_model(regular_corrupt_model, x_test_torch, y_test_torch)

        set_seed(seed)
        lre_clean_model = NN_Meta(x_fused_torch.shape[1], 2, 4, "ReLU")
        optimizer = torch.optim.SGD(lre_clean_model.params(), args.optim.lr)
        _, _, _ = train_lre(lre_clean_model, x_fused_torch, y_fused_clean_torch, x_val_torch, y_val_torch, optimizer,
                            args.optim.epochs)
        lre_clean_rates = eval_model(lre_clean_model, x_test_torch, y_test_torch)

        set_seed(seed)
        lre_corrupt_model = NN_Meta(x_fused_torch.shape[1], 2, 4, "ReLU")
        optimizer = torch.optim.SGD(lre_corrupt_model.params(), args.optim.lr)
        _, _, _ = train_lre(lre_corrupt_model, x_fused_torch, y_fused_corrupt_torch, x_val_torch, y_val_torch,
                            optimizer, args.optim.epochs)
        lre_corrupt_rates = eval_model(lre_corrupt_model, x_test_torch, y_test_torch)

        grouped_rates = {"regular": regular_rates, "regular_fused_clean": regular_clean_rates,
                         "regular_fused_corrupt": regular_corrupt_rates, "lre_fused_clean": lre_clean_rates,
                         "lre_fused_corrupt": lre_corrupt_rates}

        for train_type in grouped_rates.keys():
            for rate_name in grouped_rates[train_type].keys():
                if rate_name not in results.keys():
                    results[rate_name] = [grouped_rates[train_type][rate_name]]
                else:
                    results[rate_name].append(grouped_rates[train_type][rate_name])

            results["train_type"].append(train_type)

    csv_file_name = CSV_FILE
    data = pd.DataFrame(results)
    data.to_csv(csv_file_name, index=False, header=True)


if __name__ == "__main__":
    main()