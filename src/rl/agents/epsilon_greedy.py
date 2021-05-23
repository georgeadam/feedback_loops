import numpy as np
import torch


class EpsilonGreedyAgent:
    def __init__(self, net, eps):
        self.net = net
        self.eps = eps

    def act(self, state):
        pred = self.net(state)

        greedy = np.random.choice([0, 1], p=[self.eps, 1 - self.eps])

        if greedy:
            return torch.max(pred, 1)[1].item(), pred
        else:
            return np.random.choice(2), pred

    def q(self, state):
        return self.net(state)