import torch


def get_optimizer(name):
    if name == "sgd":
        return torch.optim.SGD
    elif name == "adam":
        return torch.optim.Adam


def create_optimizer(params, optimizer, lr, momentum, weight_decay):
    if optimizer == "SGD":
        return create_sgd(params, lr, momentum, weight_decay)
    elif optimizer == "Adam":
        return create_adam(params, lr, weight_decay)


def create_sgd(params, lr, momentum, weight_decay):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def create_adam(params, lr, weight_decay):
    return torch.optim.Adam(params, lr, weight_decay=weight_decay)