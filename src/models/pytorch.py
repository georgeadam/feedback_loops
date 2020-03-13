import torch
import torch.nn as nn

from src.utils.losses import EWC, WeightedCE
from src.utils.optimizer import get_optimizer


class StandardModel(nn.Module):
    def __init__(self, iterations=1000, tol=0.0001):
        super(StandardModel, self).__init__()

        self.iterations = iterations
        self.tol = tol
        self.device = "cuda:0"
        self.criterion = WeightedCE(self.device)

    def fit(self, x, y, weights=None):
        if self.reset_optim:
            opt = get_optimizer(self.optimizer_name)
            self.optimizer = opt(self.parameters(), lr=self.lr)

        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        x = x.to(self.device)
        y = y.to(self.device)
        no_improvement = 0
        increased_loss = 0
        best_loss = float("inf")

        for i in range(self.iterations):
            out = self.forward(x)

            loss = self.criterion(out, y, weights)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # if increased_loss >= 2:
            #     print("Loss increased for more than two iterations. Reducing LR from {} to {}".format(self.optimizer.param_groups[0]["lr"],
            #                                                                                           self.optimizer.param_groups[0]["lr"] * 0.1))
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = param_group["lr"] * 0.1

            if loss < best_loss and best_loss - loss > self.tol:
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

            if loss < self.tol or no_improvement >= 20:
                print(i)
                return loss.item()

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

        loss = self.criterion(out, y, None)

        return loss.item()

    def predict(self, x):
        x = torch.from_numpy(x).float().to(self.device)

        out = self.forward(x)

        pred = torch.max(out, 1)[1]

        return pred.cpu().numpy()

    def predict_proba(self, x):
        x = torch.from_numpy(x).float().to(self.device)

        softmax = torch.nn.Softmax(dim=1)

        out = self.forward(x)

        return softmax(out).detach().cpu().numpy()


class EWCModel(nn.Module):
    def __init__(self, iterations=1000):
        super(EWCModel, self).__init__()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.importance = 1.0
        self.iterations = iterations
        self.device = "cpu:0"

        self.ewc = None

    def fit(self, x, y):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()

        for i in range(self.iterations):
            out = self.forward(x)

            loss = self.criterion(out, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.ewc = EWC(self, x, y)

    def partial_fit(self, x, y, *args, **kwargs):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()

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


class NN(StandardModel):
    def __init__(self, num_features, iterations=1000, lr=0.01, online_lr=0.01, optimizer_name="adam", reset_optim=True,
                 tol=0.0001, hidden_layers=1, activation="relu"):
        super(NN, self).__init__(iterations=iterations, tol=tol)

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

    def forward(self, x):
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





