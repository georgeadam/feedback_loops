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


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = torch.from_numpy(prior)
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 0
        self.unlabeled = 1
        self.min_count = torch.tensor(1.)

    def forward(self, inp, probs, target, test=False):
        assert (inp.shape == target.shape)
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        if inp.is_cuda:
            self.min_count = self.min_count.to(inp.device)
            self.prior = self.prior.to(inp.device)
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count,
                                                                                            torch.sum(unlabeled))

        y_positive = self.loss_func(positive * inp) * positive
        y_positive_inv = self.loss_func(-positive * inp) * positive
        y_unlabeled = self.loss_func(-unlabeled * inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk


class PULossCombined(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, ignore_samples, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULossCombined, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = torch.from_numpy(prior)
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 0
        self.unlabeled = 1
        self.min_count = torch.tensor(1.)
        self.ignore_samples = ignore_samples

    def forward(self, inp, probs, target, test=False):
        assert (inp.shape == target.shape)

        with torch.no_grad():
            idx_range = torch.arange(len(inp)).int().to(inp.device)
            update_idx = (idx_range > self.ignore_samples)
            train_idx = (idx_range <= self.ignore_samples)

        positive, unlabeled = (target == self.positive) & update_idx, (target == self.unlabeled) & update_idx
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        if inp.is_cuda:
            self.min_count = self.min_count.to(inp.device)
            self.prior = self.prior.to(inp.device)
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count,
                                                                                            torch.sum(unlabeled))

        y_positive = self.loss_func(positive * inp) * positive
        y_positive_inv = self.loss_func(-positive * inp) * positive
        y_unlabeled = self.loss_func(-unlabeled * inp) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk + F.binary_cross_entropy(probs[train_idx], target[train_idx].float())
        else:
            return positive_risk + negative_risk + F.binary_cross_entropy(probs[train_idx], target[train_idx].float())