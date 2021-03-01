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


def load_data(data_args, model_args):
    data_fn = get_data_fn(data_args, model_args)
    x_train, y_train, x_update, y_update, x_test, y_test, cols = data_fn(data_args.n_train, data_args.n_update,
                                                                         data_args.n_test,
                                                                         num_features=data_args.num_features)
    x_update, x_val, y_update, y_val = train_test_split(x_update, y_update, test_size=0.4)
    x_val_reg, x_val_lre, y_val_reg, y_val_lre = train_test_split(x_val, y_val, test_size=0.5)

    scaler = get_scaler(True, cols)
    scaler.fit(x_train)

    x_train_torch = torch.from_numpy(scaler.transform(x_train)).float().to(model_args.device)
    x_val_reg_torch = torch.from_numpy(scaler.transform(x_val_reg)).float().to(model_args.device)
    x_val_lre_torch = torch.from_numpy(scaler.transform(x_val_lre)).float().to(model_args.device)
    x_update_torch = torch.from_numpy(scaler.transform(x_update)).float().to(model_args.device)
    x_test_torch = torch.from_numpy(scaler.transform(x_test)).float().to(model_args.device)

    y_train_torch = torch.from_numpy(y_train).long().to(model_args.device)
    y_val_reg_torch = torch.from_numpy(y_val_reg).long().to(model_args.device)
    y_val_lre_torch = torch.from_numpy(y_val_lre).long().to(model_args.device)
    y_update_torch = torch.from_numpy(y_update).long().to(model_args.device)
    y_test_torch = torch.from_numpy(y_test).long().to(model_args.device)

    data = {"x_train": x_train_torch, "y_train": y_train_torch, "x_val_reg": x_val_reg_torch, "y_val_reg": y_val_reg_torch,
            "x_val_lre": x_val_lre_torch, "y_val_lre": y_val_lre_torch, "x_update": x_update_torch, "y_update": y_update_torch,
            "x_test": x_test_torch, "y_test": y_test_torch}

    return data


def train_regular(model, x_train, y_train, x_val, y_val, optimizer, epochs, early_stopping_iter, writer, writer_prefix,
                  write=True):
    model.train()
    losses = []
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_params = copy.deepcopy(model.state_dict())
    done = False
    epoch = 0
    no_train_improvement = 0
    no_val_improvement = 0

    while not done:
        out = model(x_train)
        train_loss = F.cross_entropy(out, y_train)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(train_loss.item())

        with torch.no_grad():
            val_loss = compute_loss(model, x_val, y_val, F.cross_entropy)

        if epoch % 100 == 0:
            logger.info("Epoch: {} | Train Loss: {} | Val Loss: {}".format(epoch, train_loss.item(), val_loss.item()))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_params = copy.deepcopy(model.state_dict())
            no_val_improvement = 0
        else:
            no_val_improvement += 1

        if train_loss < best_train_loss:
            no_train_improvement = 0
        elif no_val_improvement > 0:
            no_train_improvement += 1

        if no_train_improvement > early_stopping_iter:
            done = True
            logger.info("No improvement in train loss for {} epochs at epoch: {}. Stopping.".format(early_stopping_iter,
                                                                                                    epoch))

        if no_val_improvement > early_stopping_iter:
            done = True
            logger.info("No improvement in validation loss for {} epochs at epoch: {}. Stopping.".format(early_stopping_iter,
                                                                                                         epoch))

        if epoch > epochs:
            break

        if write:
            log_regular_losses(writer, writer_prefix, train_loss.item(), val_loss.item(), epoch)

        epoch += 1

    model.load_state_dict(best_params)
    logger.info("Best Train Loss: {} | Best Val Loss : {} at epoch: {}".format(best_train_loss.item(),
                                                                               best_val_loss.item(),
                                                                               best_epoch))

    if not done:
        logger.info("Stopped after: {} epochs, but could have kept improving loss.".format(epochs))

    return losses


def train_lre(model, x_train, y_train, x_val_reg, y_val_reg, x_val_lre, y_val_lre, optimizer, model_args,
              optim_args, writer, writer_prefix, device="cpu"):
    meta_losses_clean = []
    net_losses = []
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_params = copy.deepcopy(model.state_dict())
    no_val_improvement = 0
    no_train_improvement = 0

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
        meta_model = NN_Meta(x_train.shape[1], 2, model_args.hidden_layers, model_args.activation, device)
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
        meta_model.update_params(optim_args.lr, source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_model(x_val_lre)

        l_g_meta = F.cross_entropy(y_g_hat, y_val_lre)

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

        with torch.no_grad():
            val_loss = compute_loss(model, x_val_reg, y_val_reg, F.cross_entropy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = unweighted_loss
            best_epoch = epoch
            best_params = copy.deepcopy(model.state_dict())
            no_val_improvement = 0
        else:
            no_val_improvement += 1

        if unweighted_loss < best_train_loss:
            no_train_improvement = 0
        elif no_val_improvement > 0:
            no_train_improvement += 1

        if no_train_improvement > optim_args.early_stopping_iter:
            done = True
            logger.info("No improvement in train loss for {} epochs at epoch: {}. Stopping.".format(optim_args.early_stopping_iter,
                                                                                                    epoch))

        if no_val_improvement > optim_args.early_stopping_iter:
            done = True
            logger.info("No improvement in validation loss for {} epochs at epoch: {}. Stopping.".format(optim_args.early_stopping_iter,
                                                                                                         epoch))

        if epoch % 100 == 0:
            logger.info("Epoch: {} | Weighted Train Loss: {} | Unweighted Train Loss: {} | LRE Val Loss: {} | Reg Val Loss: {}".format(epoch,
                                                                                                                                       l_f.item(),
                                                                                                                                       unweighted_loss.item(),
                                                                                                                                       l_g_meta.item(),
                                                                                                                                       val_loss.item()))

        if epoch > optim_args.epochs:
            break

        log_lre_losses(writer, writer_prefix, unweighted_loss.item(), val_loss.item(), l_g_meta.item(), epoch)

        epoch += 1

    model.load_state_dict(best_params)
    logger.info("Best Train Loss: {} | Best LRE Val Loss: {} | Best Reg Val Loss Achieved: {} at epoch: {}".format(best_train_loss.item(),
                                                                                                                   best_val_loss.item(),
                                                                                                                   l_g_meta.item(),
                                                                                                                   best_epoch))

    if not done:
        logger.info("Stopped after: {} epochs, but could have kept improving loss.".format(optim_args.epochs))

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


def compute_loss(model, x, y, criterion):
    out = model(x)

    return criterion(out, y)


def create_corrupted_labels(model, x, y):
    y_out = model(x)
    y_pred = torch.max(y_out, 1)[1]

    fps = torch.where((y == 0) & (y_pred == 1))[0]

    y_corrupt = copy.deepcopy(y)
    y_corrupt[fps] = 1
    y_corrupt_torch = y_corrupt.long().to(y.device)

    return y_corrupt_torch


def log_regular_losses(writer, writer_prefix, train_loss, val_loss, epoch):
    writer.add_scalar(writer_prefix.format(name="train_loss"), train_loss, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss"), val_loss, epoch)


def log_lre_losses(writer, writer_prefix, train_loss, val_loss_reg, val_loss_lre, epoch):
    writer.add_scalar(writer_prefix.format(name="train_loss"), train_loss, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss_reg"), val_loss_reg, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss_lre"), val_loss_lre, epoch)