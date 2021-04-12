import copy

import numpy as np
import torch
import torch.nn as nn

from src.utils.losses import EWC, WeightedCE
from src.utils.optimizer import get_optimizer


class StandardModel(nn.Module):
    def __init__(self, iterations=1000, tol=0.0001, warm_start=True):
        super(StandardModel, self).__init__()

        self.iterations = iterations
        self.tol = tol
        self.device = "cuda:0"
        self.warm_start = warm_start
        self.fitted = False

    def fit(self, x, y, sample_weight=None, log=True):
        if self.fitted and not self.warm_start:
            self.apply(weight_reset)

        if self.reset_optim:
            opt = get_optimizer(self.optimizer_name)
            self.optimizer = opt(self.parameters(), lr=self.lr)

        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()

        if self.soft:
            if type(y) == np.ndarray:
                y = torch.from_numpy(y)
            y = y.float()
        else:
            if type(y) == np.ndarray:
                y = torch.from_numpy(y)
            y = y.long()

        if self.soft:
            y_copy = copy.deepcopy(y)
            y_copy[y_copy == 1] = 0.9
            y_copy[y_copy == 0] = 0.1

            y_one_hot = torch.FloatTensor(len(y), 2)
            y_one_hot.zero_()
            y_one_hot[:, 1] = y_copy
            y_one_hot[:, 0] = 1 - y_copy
            y = y_one_hot

        x = x.to(self.device)
        y = y.to(self.device)
        no_improvement = 0
        increased_loss = 0
        best_loss = float("inf")
        best_params = None

        for i in range(self.iterations):
            out = self.forward(x)

            loss = self.criterion(out, y, sample_weight)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if loss < best_loss and best_loss - loss > self.tol:
                best_params = self.state_dict()
                best_loss = loss.item()
                no_improvement = 0
                increased_loss = 0
            elif loss > best_loss:
                if log:
                    print("Loss increased!")
                increased_loss += 1
                no_improvement += 1
            else:
                no_improvement += 1
                increased_loss = 0

            if loss < self.tol:
                print("Loss lower than desired tolerance at iteration: {}. Stopping Early".format(i))
                break
            elif no_improvement >= 50:
                print("No improvement in loss for 50 iterations at iteration: {}. Stopping Early".format(i))
                print("Final loss is: {}".format(loss))
                break

        if best_params is not None:
            self.load_state_dict(best_params)

        self.fitted = True

        return loss.item()

    def partial_fit(self, x, y, weights=None, *args, **kwargs):
        if self.reset_optim:
            opt = get_optimizer(self.optimizer_name)
            self.optimizer = opt(self.parameters(), lr=self.online_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.online_lr

        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.forward(x)

        loss = self.criterion(out, y, weights)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def evaluate(self, x, y):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        x = x.to(self.device)
        y = y.to(self.device)
        out = self.forward(x)

        # loss = self.criterion(out, y, None)

        return 0.0

    def predict(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)

        out = self.forward(x)

        pred = torch.max(out, 1)[1]

        return pred.cpu().numpy()

    def predict_proba(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)

        softmax = torch.nn.Softmax(dim=1)

        out = self.forward(x)

        return softmax(out).detach().cpu().numpy()


class EWCModel(nn.Module):
    def __init__(self, importance=0.1, iterations=1000, tol=0.0001):
        super(EWCModel, self).__init__()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.importance = importance
        self.iterations = iterations
        self.tol = tol
        self.device = "cuda:0"

        self.ewc = None

    def fit(self, x, y, sample_weight=None):
        x, y = torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).long().to(self.device)
        no_improvement = 0
        increased_loss = 0
        best_loss = float("inf")
        best_params = None

        if self.ewc is None:
            for i in range(self.iterations):
                out = self.forward(x)

                loss = self.criterion(out, y)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if loss < best_loss and best_loss - loss > self.tol:
                    best_params = self.state_dict()
                    best_loss = loss.item()
                    no_improvement = 0
                    increased_loss = 0
                elif loss > best_loss:
                    print("Loss increased!")
                    increased_loss += 1
                    no_improvement += 1
                else:
                    no_improvement += 1
                    increased_loss = 0

                if loss < self.tol:
                    print("Loss lower than desired tolerance at iteration: {}. Stopping Early".format(i))
                    break
                elif no_improvement >= 50:
                    print("No improvement in loss for 50 iterations at iteration: {}. Stopping Early".format(i))
                    break

            if best_params is not None:
                self.load_state_dict(best_params)

            self.ewc = EWC(self, x, y)
        else:
            for i in range(self.iterations):
                out = self.forward(x)

                ce_loss = self.criterion(out, y)
                penalty = self.importance * self.ewc.penalty(self)

                loss = ce_loss + penalty
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if loss < best_loss and best_loss - loss > self.tol:
                    best_params = self.state_dict()
                    best_loss = loss.item()
                    no_improvement = 0
                    increased_loss = 0
                elif loss > best_loss:
                    print("Loss increased!")
                    increased_loss += 1
                    no_improvement += 1
                else:
                    no_improvement += 1
                    increased_loss = 0

                if loss < self.tol:
                    print("Loss lower than desired tolerance at iteration: {}. Stopping Early".format(i))
                    break
                elif no_improvement >= 50:
                    print("No improvement in loss for 50 iterations at iteration: {}. Stopping Early".format(i))
                    break

            if best_params is not None:
                self.load_state_dict(best_params)

    def partial_fit(self, x, y, *args, **kwargs):
        x, y = torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).long().to(self.device)

        out = self.forward(x)

        ce_loss = self.criterion(out, y)
        penalty = self.importance * self.ewc.penalty(self)

        loss = ce_loss + penalty

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def predict(self, x):
        x = torch.from_numpy(x).float().to(self.device)

        out = self.forward(x)

        pred = torch.max(out, 1)[1]

        return pred.detach().cpu().numpy()

    def predict_proba(self, x):
        x = torch.from_numpy(x).float().to(self.device)

        softmax = torch.nn.Softmax(dim=1)

        out = self.forward(x)

        return softmax(out).detach().cpu().numpy()

    def evaluate(self, x, y):
        return 0.0


class DeferNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, reject_neuron=False, output_method="softmax"):
        super(DeferNN, self).__init__()
        self.output_method = output_method
        self.reject_neuron = reject_neuron

        if reject_neuron:
            self.fc = torch.nn.Linear(input_dim, output_dim + 1)
        else:
            self.fc = torch.nn.Linear(input_dim, output_dim)

        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, softmax=True):
        x = self.fc(x)

        if softmax:
            if self.output_method == "softmax":
                x = self.softmax(x)
            else:
                x = torch.cat([self.softmax(x[:, :x.shape[1] - 1]), self.sigmoid(x[:, -1]).view(-1, 1)], dim=1)

        return x

    def fit(self, x, y, epochs, lr=0.1, logging=True):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            out = self.forward(x, softmax=False)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0 and logging:
                print("Epoch: {} | Loss: {}".format(epoch, loss.item()))

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
            if self.reject_neuron and self.output_method == "sigmoid":
                _, predicted = torch.max(out, 1)
                reject = out[:, -1] > 0.5
                predicted[reject] = out.shape[1] - 1
            else:
                _, predicted = torch.max(out, 1)

        return predicted.detach().numpy()


class DeferDeepNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, reject_neuron=False, output_method="softmax"):
        super(DeferDeepNN, self).__init__()
        self.output_method = output_method
        self.reject_neuron = reject_neuron

        if reject_neuron:
            self.fc1 = torch.nn.Linear(input_dim, 3)
            self.fc2 = torch.nn.Linear(3, output_dim + 1)
        else:
            self.fc1 = torch.nn.Linear(input_dim, 3)
            self.fc2 = torch.nn.Linear(3, output_dim)

        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x, softmax=True):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        if softmax:
            if self.output_method == "softmax":
                x = self.softmax(x)
            else:
                x = torch.cat([self.softmax(x[:, :x.shape[1] - 1]), self.sigmoid(x[:, -1]).view(-1, 1)], dim=1)

        return x

    def fit(self, x, y, epochs, lr=0.1, logging=True):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            out = self.forward(x, softmax=False)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0 and logging:
                print("Epoch: {} | Loss: {}".format(epoch, loss.item()))

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
            if self.reject_neuron and self.output_method == "sigmoid":
                _, predicted = torch.max(out, 1)
                reject = out[:, -1] > 0.5
                predicted[reject] = out.shape[1] - 1
            else:
                _, predicted = torch.max(out, 1)

        return predicted.detach().numpy()


class NNEWC(EWCModel):
    def __init__(self, num_features, iterations=1000, lr=0.01, online_lr=0.01, optimizer_name="adam", reset_optim=True,
                 tol=0.0001, hidden_layers=1, activation="relu", soft=False, importance=1.0):
        super(NNEWC, self).__init__(iterations=iterations)

        self.num_features = num_features
        self.device = "cuda:0"
        self.fc = None
        self.lr = lr
        self.online_lr = online_lr
        self.optimizer_name = optimizer_name
        self.reset_optim = reset_optim
        self._create_layers(hidden_layers)

        self.activation = getattr(nn, activation)()

        opt = get_optimizer(self.optimizer_name)
        self.optimizer = opt(self.parameters(), lr=self.lr)
        self.soft = soft
        # self.criterion = WeightedCE(self.device, soft)

        self.importance = importance

    def forward(self, x):
        x = x.to(self.device)

        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.activation(x)

        x = self.fc[-1](x)

        return x

    def _create_layers(self, hidden_layers):
        if hidden_layers == 0:
            self.fc = nn.ModuleList([nn.Linear(self.num_features, 2).to(self.device)])
        elif hidden_layers == 1:
            self.fc = nn.ModuleList([nn.Linear(self.num_features, 10).to(self.device),
                                    nn.Linear(10, 2).to(self.device)])
        elif hidden_layers == 2:
            self.fc = nn.ModuleList([nn.Linear(self.num_features, 20).to(self.device),
                                     nn.Linear(20, 10).to(self.device),
                                     nn.Linear(10, 2).to(self.device)])


class ReweightNN(torch.nn.Module):
    def __init__(self, num_features, iterations=1000, lr=0.01, online_lr=0.01, optimizer_name="adam", reset_optim=True,
                 tol=0.0001, hidden_layers=1, activation="relu", soft=False):
        super(ReweightNN, self).__init__()

        self.num_features = num_features
        self.device = "cuda:0"
        self.fc = None
        self.lr = lr
        self.iterations = iterations
        self.online_lr = online_lr
        self.optimizer_name = optimizer_name
        self.reset_optim = reset_optim
        self.tol = tol
        self.hidden_layers = hidden_layers
        self._create_layers(hidden_layers)

        self.activation = getattr(nn, activation)()

        opt = get_optimizer(self.optimizer_name)
        self.optimizer = opt(self.parameters(), lr=self.lr)
        self.soft = soft

    def fit(self, x, y):
        x, y = torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).long().to(self.device)
        meta_losses = []
        net_losses = []

        meta_loss = 0
        net_loss = 0

        meta_net = ReweightNN(self.num_features, self.iterations, self.lr, self.online_lr, self.optimizer_name,
                              self.reset_optim, self.tol, self.hidden_layers, self.activation, self.soft)
        meta_net = meta_net.to(self.device)

        for i in range(self.iterations):
            meta_net.load_state_dict(self.state_dict())

            y_f_hat = meta_net(x)
            cost = torch.nn.functional.binary_cross_entropy_loss_with_logits(y_f_hat, y, reduce=False)
            eps = torch.zeros(len(cost)).to(cost.device)
            l_f_meta = torch.sum(eps * cost)

            meta_net.zero_grad()

            grads = torch.autograd.grad(l_f_meta, meta_net.parameters(), create_graph=True)
            meta_net.update_params(self.lr, source_params=grads)

            y_g_hat = meta_net,

    def forward(self, x):
        x = x.to(self.device)

        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.activation(x)

        x = self.fc[-1](x)

        return x

    def _create_layers(self, hidden_layers):
        if hidden_layers == 0:
            self.fc = nn.ModuleList([nn.Linear(self.num_features, 2).to(self.device)])
        elif hidden_layers == 1:
            self.fc = nn.ModuleList([nn.Linear(self.num_features, 10).to(self.device),
                                    nn.Linear(10, 2).to(self.device)])
        elif hidden_layers == 2:
            self.fc = nn.ModuleList([nn.Linear(self.num_features, 20).to(self.device),
                                     nn.Linear(20, 10).to(self.device),
                                     nn.Linear(10, 2).to(self.device)])


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)

    if callable(reset_parameters):
        m.reset_parameters()
