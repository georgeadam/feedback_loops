import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class EWC(object):
    def __init__(self, model: nn.Module, x, y):

        self.model = model
        self.x = x
        self.y = y


        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()

        self.model.zero_grad()


        output = self.model(self.x)
        loss = F.nll_loss(F.log_softmax(output, dim=1), self.y)
        loss.backward()

        for n, p in self.model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class WeightedCE(object):
    def __init__(self, device, soft=False):
        if soft:
            self.ce = MultiClassCE(reduction="none")
        else:
            self.ce = torch.nn.CrossEntropyLoss(reduction="none")

        self.device = device

    def __call__(self, input, target, weight):
        loss = self.ce(input, target)

        if weight is not None and len(weight) == len(input):
            loss = torch.mean(loss * torch.from_numpy(weight).float().to(self.device))
        else:
            loss = torch.mean(loss)

        return loss


class MultiClassCE(object):
    def __init__(self, reduction="none"):
        self.logsoftmax = torch.nn.LogSoftmax(1)
        self.reduction = reduction

    def __call__(self, predictions, targets):
        if self.reduction == "none":
            return torch.sum(- targets * self.logsoftmax(predictions), 1)
        else:
            return torch.mean(torch.sum(- targets * self.logsoftmax(predictions), 1))