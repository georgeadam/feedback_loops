import torch
import torch.nn as nn

from src.utils.losses import EWC


class StandardModel(nn.Module):
    def __init__(self, iterations=1000):
        super(StandardModel, self).__init__()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.iterations = iterations
        self.device = "cpu:0"

    def fit(self, x, y):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        x = x.to(self.device)
        y = y.to(self.device)

        for i in range(self.iterations):
            out = self.forward(x)

            loss = self.criterion(out, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

    def partial_fit(self, x, y, *args, **kwargs):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.forward(x)

        loss = self.criterion(out, y)
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

        softmax = torch.nn.Softmax()

        out = self.forward(x)

        return softmax(out / 10.0).detach().cpu().numpy()

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

        self.iterations = iterations
        self.device = "cpu:0"

        self.x = None
        self.y = None
        self.ewc = None

    def fit(self, x, y):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
        self.x = x
        self.y = y
        self.ewc = EWC(self, self.x, self.y)

        for i in range(self.iterations):
            out = self.forward(x)

            loss = self.criterion(out, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

    def partial_fit(self, x, y, *args, **kwargs):
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()

        out = self.forward(x)

        loss = self.criterion(out, y) + self.ewc.penalty(self)
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

        softmax = torch.nn.Softmax()

        out = self.forward(x)

        return softmax(out / 10.0).detach().cpu().numpy()


class LREWC(EWCModel):
    def __init__(self, num_features, iterations=1000, lr=1.0):
        super(LREWC, self).__init__(iterations=iterations)

        self.w = nn.Linear(num_features, 2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        return self.w(x)


class NN(StandardModel):
    def __init__(self, num_features, iterations=1000, lr=0.01):
        super(NN, self).__init__(iterations=iterations)

        self.device = "cuda:0"
        self.fc1 = nn.Linear(num_features, 50).to(self.device)
        self.fc2 = nn.Linear(50, num_features).to(self.device)

        self.activation = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)

        return x



