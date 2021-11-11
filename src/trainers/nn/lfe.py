import copy
import itertools
import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils.str_formatting import SafeDict
from src.utils.optimizer import create_optimizer
from src.data.wrappers import DataMiniBatcher
from src.trainers.nn.utils import compute_loss
from src.utils.train import train_regular_nn

logger = logging.getLogger(__name__)


class LFENNTrainer:
    def __init__(self, model_fn, seed, warm_start, update, regular_optim_args, lfe_optim_args, **kwargs):
        self._warm_start = warm_start
        self._update = update
        self._model_fn = model_fn

        self._epochs_regular = regular_optim_args.epochs
        self._early_stopping_iter_regular = regular_optim_args.early_stopping_iter
        self._optimizer_name_regular = regular_optim_args.optimizer
        self._lr_regular = regular_optim_args.lr
        self._momentum_regular = regular_optim_args.momentum
        self._nesterov_regular = regular_optim_args.nesterov
        self._weight_decay_regular = regular_optim_args.weight_decay
        self._fit_type_regular = regular_optim_args.fit_type

        self._epochs_lfe = lfe_optim_args.epochs
        self._early_stopping_iter_lfe = lfe_optim_args.early_stopping_iter
        self._optimizer_name_lfe = lfe_optim_args.optimizer
        self._lr_lfe = lfe_optim_args.lr
        self._momentum_lfe = lfe_optim_args.momentum
        self._nesterov_lfe = lfe_optim_args.nesterov
        self._weight_decay_lfe = lfe_optim_args.weight_decay
        self._incremental = lfe_optim_args.incremental
        self._label_update_fn = self._init_label_update_fn(lfe_optim_args.label_update_args)
        self._eps_penalty = lfe_optim_args.label_update_args.eps_penalty

        self._optimizer_lre = None
        self._writer_prefix = "{type}/{update_num}/{name}"
        self._write = regular_optim_args.log_tensorboard
        self._seed = seed

        if self._write:
            self._writer = SummaryWriter("tensorboard_logs/{}".format(seed))
        else:
            self._writer = None

    def _init_label_update_fn(self, label_update_args):
        if label_update_args.type == "y_direct":
            return wrapped(YDirectUpdater, momentum=label_update_args.momentum)
        elif label_update_args.type == "y_sign":
            return wrapped(YSignUpdater, momentum=label_update_args.momentum)
        elif label_update_args.type == "y_eps":
            return wrapped(YEpsUpdater, momentum=label_update_args.momentum)

    def initial_fit(self, model, data_wrapper, scaler):
        if self._fit_type_regular == "regular":
            self._regular_fit_init(model, data_wrapper, scaler)
        else:
            self._lfe_fit_init(model, data_wrapper, scaler)

    def _regular_fit_init(self, model, data_wrapper, scaler):
        optimizer = create_optimizer(model.params(), self._optimizer_name_regular,
                                     self._lr_regular, self._momentum_regular,
                                     self._nesterov_regular, self._weight_decay_regular)

        x_train, y_train = data_wrapper.get_init_train_data()
        x_val_lre, y_val_lre = data_wrapper.get_val_data_lre()
        x_val_reg, y_val_reg = data_wrapper.get_val_data_regular()

        x_train, x_val_lre, x_val_reg = scaler.transform(x_train), \
                                        scaler.transform(x_val_lre), \
                                        scaler.transform(x_val_reg)

        train_regular_nn(model, optimizer, F.cross_entropy, x_train, y_train, x_val_reg, y_val_reg,
                         self._epochs_regular, self._early_stopping_iter_regular, self._writer,
                         self._writer_prefix.format_map(SafeDict(type="regular", update_num=0)), self._write)

    def _lfe_fit_init(self, model, data_wrapper, scaler):
        optimizer = create_optimizer(model.params(), self._optimizer_name_lfe,
                                     self._lr_lfe, self._momentum_lfe,
                                     self._nesterov_lfe, self._weight_decay_lfe)

        x_train, y_train = data_wrapper.get_init_train_data()
        x_val_lfe, y_val_lfe = data_wrapper.get_val_data_lre()
        x_val_reg, y_val_reg = data_wrapper.get_val_data_regular()

        x_train, x_val_lfe, x_val_reg = scaler.transform(x_train), \
                                        scaler.transform(x_val_lfe), \
                                        scaler.transform(x_val_reg)

        y_train = y_train.copy()
        label_updater = self._label_update_fn()

        y_corrected = train_lfe(model, self._model_fn, x_train, y_train, x_val_reg, y_val_reg, x_val_lfe, y_val_lfe,
                                data_wrapper._batch_size, optimizer,
                                self._lr_lfe, self._epochs_lfe, self._early_stopping_iter_lfe, self._incremental, label_updater, self._eps_penalty,
                                self._writer, self._writer_prefix.format_map(SafeDict(type="lre", update_num=0)), self._write, 0,
                                self._seed)

        logger.info("**********y_train_corrupted**********")
        logger.info(y_train)

        logger.info("**********y_train_clean**********")
        logger.info(data_wrapper._y_train_clean)

        logger.info("**********y_corrected**********")
        logger.info(y_corrected)

        idx = data_wrapper._y_train_clean == y_train
        logger.info("**********y_train[y_train_clean == y_train_corrupted]*********")
        logger.info(y_train[idx])

        logger.info("*********y_corrected[y_train_clean == y_train_corrupted]*********")
        logger.info(y_corrected[idx])

        with open("y_train_corrupted{}_{}.npy".format(self._seed, 0), "wb") as f:
            np.save(f, y_train)

        with open("y_train_clean{}_{}.npy".format(self._seed, 0), "wb") as f:
            np.save(f, data_wrapper._y_train_clean)

        with open("y_corrected{}_{}.npy".format(self._seed, 0), "wb") as f:
            np.save(f, y_corrected.detach().cpu().numpy())

    def update_fit(self, model, data_wrapper, rate_tracker, scaler, update_num, *args):
        if not self._update:
            return model

        if not self._warm_start:
            model = self._model_fn(data_wrapper.dimension).to(model.device)
            self._optimizer_lre = create_optimizer(model.params(), self._optimizer_name_lfe,
                                                   self._lr_lfe, self._momentum_lfe,
                                                   self._nesterov_lfe, self._weight_decay_lfe)
        elif self._optimizer_lre is None:
            self._optimizer_lre = create_optimizer(model.params(), self._optimizer_name_lfe,
                                                   self._lr_lfe, self._momentum_lfe,
                                                   self._nesterov_lfe, self._weight_decay_lfe)

        x_train_corrupt, y_train_corrupt = data_wrapper.get_all_data_for_model_fit_corrupt()

        x_val_lre, y_val_lre = data_wrapper.get_val_data_lre()
        x_val_reg, y_val_reg = data_wrapper.get_val_data_regular()

        x_train_corrupt, x_val_lre, x_val_reg = scaler.transform(x_train_corrupt), \
                                                scaler.transform(x_val_lre), \
                                                scaler.transform(x_val_reg)

        y_train_corrupt = y_train_corrupt.copy()
        label_updater = self._label_update_fn()

        y_corrected = train_lfe(model, self._model_fn, x_train_corrupt, y_train_corrupt, x_val_reg, y_val_reg, x_val_lre, y_val_lre,
                                data_wrapper._batch_size, self._optimizer_lre,
                                self._lr_lfe, self._epochs_lfe, self._early_stopping_iter_lfe, self._incremental, label_updater, self._eps_penalty,
                                self._writer, self._writer_prefix.format_map(SafeDict(type="lfe", update_num=update_num)),
                                self._write,
                                update_num, self._seed)

        return model


def train_lfe(model, model_fn, x_train, y_train, x_val_reg, y_val_reg, x_val_lre, y_val_lre, batch_size, optimizer,
              lr, epochs, early_stopping_iter, incremental, label_updater, eps_penalty, writer, writer_prefix, write, update_num, seed):
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_params = copy.deepcopy(model.state_dict())
    no_val_improvement = 0
    no_train_improvement = 0

    done = False
    epoch = 0

    x_train = torch.from_numpy(x_train).float().to(model.device)
    y_train = torch.from_numpy(y_train).float().to(model.device)
    y_train.requires_grad = True

    x_val_lre = torch.from_numpy(x_val_lre).float().to(model.device)
    y_val_lre = torch.from_numpy(y_val_lre).float().to(model.device)

    x_val_reg = torch.from_numpy(x_val_reg).float().to(model.device)
    y_val_reg = torch.from_numpy(y_val_reg).long().to(model.device)

    train_data_loader = DataMiniBatcher(x_train, y_train, batch_size)
    val_lre_data_loader = DataMiniBatcher(x_val_lre, y_val_lre, batch_size)

    if len(x_train) < len(x_val_lre):
        train_data_loader = itertools.cycle(train_data_loader)
    else:
        val_lre_data_loader = itertools.cycle(val_lre_data_loader)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    v = None

    while not done:
        y_train.requires_grad = True

        weighted_train_losses = []
        unweighted_train_losses = []
        meta_losses = []

        model.train()
        # Line 2 get batch of data
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_model = model_fn(x_train.shape[1]).to(model.device)
        meta_model.load_state_dict(model.state_dict())
        meta_model = meta_model.to(x_train.device)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_model(x_train)
        eps = torch.zeros(y_train.size()).to(y_train.device).requires_grad_(True)
        cost = loss_fn(y_f_hat[:, 1], y_train + eps)
        cost = torch.mean(cost)

        meta_model.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(cost, (meta_model.params()), create_graph=True)
        meta_model.update_params(lr, source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_model(x_val_lre)

        l_g_meta = loss_fn(y_g_hat[:, 1], y_val_lre)
        l_g_meta = torch.mean(l_g_meta)

        y_train = label_updater.update_y(l_g_meta, y_train, eps)

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = model(x_train)
        cost = loss_fn(y_f_hat[:, 1], y_train.detach())
        l_f = torch.mean(cost)

        if torch.sum(label_updater.grad_eps > 0):
            l_f += eps_penalty * torch.norm(label_updater.grad_eps, p=1)

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        with torch.no_grad():
            unweighted_loss = loss_fn(y_f_hat[:, 1], y_train.detach())
            unweighted_loss = torch.mean(unweighted_loss)

        unweighted_train_losses.append(unweighted_loss.item())
        weighted_train_losses.append(l_f.detach().item())
        meta_losses.append(l_g_meta.detach().item())

        unweighted_train_loss = np.mean(np.array(unweighted_train_losses))
        weighted_train_loss = np.mean(np.array(weighted_train_losses))
        meta_loss = np.mean(np.array(meta_losses))

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

        if unweighted_train_loss < best_train_loss:
            no_train_improvement = 0
        elif no_val_improvement > 0:
            no_train_improvement += 1

        if no_train_improvement > early_stopping_iter:
            done = True
            logger.info("No improvement in train loss for {} epochs at epoch: {}. Stopping.".format(early_stopping_iter,
                                                                                                    epoch))

        if no_val_improvement > early_stopping_iter:
            done = True
            logger.info(
                "No improvement in validation loss for {} epochs at epoch: {}. Stopping.".format(early_stopping_iter,
                                                                                                 epoch))

        if epoch % 100 == 0:
            logger.info(
                "Epoch: {} | Weighted Train Loss: {} | Unweighted Train Loss: {} | LFE Val Loss: {} | Reg Val Loss: {}".format(
                    epoch,
                    weighted_train_loss,
                    unweighted_train_loss,
                    meta_loss,
                    val_loss.item()))

        if epoch > epochs:
            break

        if write:
            log_lfe_losses(writer, writer_prefix, unweighted_train_loss, val_loss.item(), meta_loss, epoch)

        epoch += 1

    # with open("sample_weights_{}_{}.npy".format(seed, update_num), "wb") as f:
    #     np.save(f, np.array(w_history))

    model.load_state_dict(best_params)
    logger.info("Best Train Loss: {} | Best LFE Val Loss: {} | Best Reg Val Loss Achieved: {} at epoch: {}".format(
        best_train_loss.item(),
        best_val_loss.item(),
        meta_loss,
        best_epoch))

    if not done:
        logger.info("Stopped after: {} epochs, but could have kept improving loss.".format(epochs))

    return y_train


def wrapped(label_update_fn, **kwargs):
    def inner():
        return label_update_fn(**kwargs)

    return inner


class YDirectUpdater():
    def __init__(self, momentum, *args, **kwargs):
        self.momentum = momentum
        self.v = None
        self.grad_eps = None

    def update_y(self, loss, y, *args):
        grad_y = torch.autograd.grad(loss, y, only_inputs=True)[0]

        if self.momentum:
            if self.v is None:
                self.v = grad_y
            else:
                self.v += grad_y
        else:
            self.v = grad_y

        with torch.no_grad():
            updated_y = torch.clamp(y - self.v, 0, 1)

        self.grad_eps = torch.zeros(y.shape).to(y.device)

        return updated_y


class YSignUpdater():
    def __init__(self, momentum, *args, **kwargs):
        self.momentum = momentum
        self.v = None
        self.grad_eps = None

    def update_y(self, loss, y, *args):
        grad_y = torch.autograd.grad(loss, y, only_inputs=True)[0]
        grad_y = torch.sign(grad_y) * 0.01

        if self.momentum:
            if self.v is None:
                self.v = grad_y
            else:
                self.v += grad_y
        else:
            self.v = grad_y

        with torch.no_grad():
            updated_y = torch.clamp(y - self.v, 0, 1)

        self.grad_eps = torch.zeros(y.shape).to(y.device)

        return updated_y


class YEpsUpdater():
    def __init__(self, momentum):
        self.momentum = momentum
        self.v = None
        self.grad_eps = None

    def update_y(self, loss, y, eps):
        grad_eps = torch.autograd.grad(loss, eps, only_inputs=True)[0]

        if self.momentum:
            if self.v is None:
                self.v = grad_eps
            else:
                self.v += grad_eps
        else:
            self.v = grad_eps

        self.grad_eps = grad_eps

        with torch.no_grad():
            updated_y = torch.clamp(y - self.v, 0, 1)

        return updated_y


class YSigmoidAnnealing():
    def __init__(self, momentum):
        self.momentum = momentum
        self.v = None
        self.grad_eps = None

    def update_y(self, loss, y, eps, epochs, epoch):
        grad_y = torch.autograd.grad(loss, y, only_inputs=True)[0]

        if self.momentum:
            if self.v is None:
                self.v = grad_y
            else:
                self.v += grad_y
        else:
            self.v = grad_y

        with torch.no_grad():
            updated_y = torch.clamp(y - self.v, 0, 1)

        self.grad_eps = torch.zeros(y.shape).to(y.device)

        return updated_y


def log_lfe_losses(writer, writer_prefix, train_loss, val_loss_reg, val_loss_lre, epoch):
    writer.add_scalar(writer_prefix.format(name="train_loss"), train_loss, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss_reg"), val_loss_reg, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss_lfe"), val_loss_lre, epoch)


def cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax()

    return torch.sum(- soft_targets * logsoftmax(pred), 1)