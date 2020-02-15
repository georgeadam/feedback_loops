import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, input_dim, num_classes, iterations=1000, lr=0.001):
        super(LR, self).__init__()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.iterations = iterations
        self.w = nn.Linear(input_dim, num_classes)

        self.optimizer = torch.optim.SGD(self.w, lr=lr)

    def forward(self, x):
        return self.w(x)

    def fit(self, x, y):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        for i in range(self.iterations):
            out = self.forward(x)

            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def partial_fit(self, x, y):
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        out = self.forward(x)

        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()