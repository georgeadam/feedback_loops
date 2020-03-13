import torch


def get_optimizer(name):
    if name == "sgd":
        return torch.optim.SGD
    elif name == "adam":
        return torch.optim.Adam