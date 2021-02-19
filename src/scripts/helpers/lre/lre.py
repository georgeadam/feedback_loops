import copy
import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from src.models.lre import NN_Meta
from src.utils.data import get_data_fn
from src.utils.metrics import compute_all_rates
from src.utils.preprocess import get_scaler

logger = logging.getLogger(__name__)


def load_data(data, model):
    data_fn = get_data_fn(data, model)
    x_train, y_train, x_update, y_update, x_test, y_test, cols = data_fn(data.n_train, data.n_update, data.n_test, num_features=data.num_features)
    x_update, x_val, y_update, y_val = train_test_split(x_update, y_update, test_size=0.05)

    scaler = get_scaler(True, cols)
    scaler.fit(x_train)

    x_train_torch = torch.from_numpy(scaler.transform(x_train)).float().to(model.device)
    x_val_torch = torch.from_numpy(scaler.transform(x_val)).float().to(model.device)
    x_update_torch = torch.from_numpy(scaler.transform(x_update)).float().to(model.device)
    x_test_torch = torch.from_numpy(scaler.transform(x_test)).float().to(model.device)

    y_train_torch = torch.from_numpy(y_train).long().to(model.device)
    y_val_torch = torch.from_numpy(y_val).long().to(model.device)
    y_update_torch = torch.from_numpy(y_update).long().to(model.device)
    y_test_torch = torch.from_numpy(y_test).long().to(model.device)

    return (x_train_torch, y_train_torch, x_val_torch, y_val_torch,
            x_update_torch, y_update_torch, x_test_torch, y_test_torch)


def train_regular(model, x, y, optimizer, epochs, tol):
    model.train()
    losses = []
    best_loss = float("inf")
    best_epoch = 0
    best_params = copy.deepcopy(model.state_dict())
    done = False
    epoch = 0

    while not done:
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if epoch % 100 == 0:
            logger.info("Epoch: {} | Loss: {}".format(epoch, loss.item()))

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_params = copy.deepcopy(model.state_dict())

        if loss < tol:
            done = True
            logger.info("Converged to desired loss of: {} or less at epoch: {}".format(tol, epoch))

        if epoch > epochs:
            break

        epoch += 1

    model.load_state_dict(best_params)
    logger.info("Best Loss Achieved: {} at epoch: {}".format(best_loss, best_epoch))

    if not done:
        logger.info("Failed to converge to desired loss: {} after {} epochs".format(tol, epochs))

    return losses


def train_lre(model, x_train, y_train, x_val, y_val, optimizer, args, device="cpu"):
    meta_losses_clean = []
    net_losses = []
    best_loss = float("inf")
    best_epoch = 0
    best_params = copy.deepcopy(model.state_dict())

    smoothing_alpha = 0.9
    meta_l = 0
    net_l = 0
    done = False
    epoch = 0

    while not done:
        model.train()
        # Line 2 get batch of data
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_model = NN_Meta(x_train.shape[1], 2, args.model.hidden_layers, args.model.activation, device)
        meta_model.load_state_dict(model.state_dict())
        meta_model = meta_model.to(x_train.device)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_model(x_train)
        cost = F.cross_entropy(y_f_hat, y_train, reduce=False)
        eps = torch.zeros(cost.size()).to(x_train.device).requires_grad_(True)

        l_f_meta = torch.sum(cost * eps)

        meta_model.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_model.update_params(args.optim.lre_lr, source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_model(x_val)

        l_g_meta = F.cross_entropy(y_g_hat, y_val)

        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = model(x_train)
        cost = F.cross_entropy(y_f_hat, y_train, reduce=False)
        l_f = torch.sum(cost * w)

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        with torch.no_grad():
            unweighted_loss = F.cross_entropy(y_f_hat, y_train)

        meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
        meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (epoch + 1)))

        net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
        net_losses.append(net_l / (1 - smoothing_alpha ** (epoch + 1)))

        if unweighted_loss < best_loss:
            best_loss = unweighted_loss
            best_epoch = epoch
            best_params = copy.deepcopy(model.state_dict())

        if epoch % 100 == 0:
            logger.info("Epoch: {} | Weighted Train Loss: {} | Unweighted Train Loss: {} | Val Loss: {}".format(epoch,
                                                                                                                l_f.item(),
                                                                                                                unweighted_loss.item(),
                                                                                                                l_g_meta.item()))

        if unweighted_loss < args.optim.lre_tol:
            done = True
            logger.info("Converged to desired loss of: {} or less at epoch: {}".format(args.optim.lre_tol, epoch))

        if epoch > args.optim.epochs:
            break

        epoch += 1

    model.load_state_dict(best_params)
    logger.info("Best Loss Achieved: {} at epoch: {}".format(best_loss, best_epoch))

    if not done:
        logger.info("Failed to converge to desired loss: {} after {} epochs".format(args.optim.lre_tol, args.optim.epochs))

    return meta_losses_clean, net_losses, w


def eval_model(model, x, y):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    y_out = model(x)
    y_prob = softmax(y_out)
    y_pred = torch.max(y_out, 1)[1]

    rates = compute_all_rates(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy(),
                                        y_prob.detach().cpu().numpy())
    return rates


def create_corrupted_labels(model, x, y):
    y_out = model(x)
    y_pred = torch.max(y_out, 1)[1]

    fps = torch.where((y == 0) & (y_pred == 1))[0]

    y_corrupt = copy.deepcopy(y)
    y_corrupt[fps] = 1
    y_corrupt_torch = y_corrupt.long().to(y.device)

    return y_corrupt_torch