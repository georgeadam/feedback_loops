import torch
import torch.nn as nn

from src.utils.losses import EWC, WeightedCE


class StandardModel(nn.Module):
    def __init__(self, iterations=1000):
        super(StandardModel, self).__init__()

        self.iterations = iterations
        self.device = "cuda:0"
        self.criterion = WeightedCE(self.device)

    def fit(self, x, y, weights=None):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        x = x.to(self.device)
        y = y.to(self.device)

        for i in range(self.iterations):
            out = self.forward(x)

            loss = self.criterion(out, y, weights)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

    def partial_fit(self, x, y, weights=None, *args, **kwargs):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.forward(x)

        loss = self.criterion(out, y, weights)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

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


class LR(StandardModel):
    def __init__(self, num_features, iterations=1000, lr=1.0):
        super(LR, self).__init__(iterations=iterations)

        self.w = nn.Linear(num_features, 2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        return self.w(x)


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


class LREWC(EWCModel):
    def __init__(self, num_features, iterations=1000, lr=0.1, importance=1.0):
        super(LREWC, self).__init__(iterations=iterations)

        self.importance = importance
        self.w = nn.Linear(num_features, 2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        return self.w(x)


class NN(StandardModel):
    def __init__(self, num_features, iterations=1000, lr=0.01, hidden_layers=1, activation="relu"):
        super(NN, self).__init__(iterations=iterations)

        self.num_features = num_features
        self.device = "cuda:0"
        self.fc = None
        self._create_layers(hidden_layers)

        self.activation = getattr(nn, activation)()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)


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





