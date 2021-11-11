import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils.losses import PULoss, PULossCombined
from src.trainers.nn.utils import log_regular_losses
from src.utils.optimizer import create_optimizer
from src.utils.train import train_regular_nn

logger = logging.getLogger(__name__)


class PUNNTrainer:
    def __init__(self, model_fn, seed, warm_start, update, optim_args, **kwargs):
        self._warm_start = warm_start
        self._update = update
        self._epochs = optim_args.epochs
        self._early_stopping_iter = optim_args.early_stopping_iter
        self._model_fn = model_fn

        self._optimizer_name = optim_args.optimizer
        self._lr = optim_args.lr
        self._momentum = optim_args.momentum
        self._nesterov = optim_args.nesterov
        self._weight_decay = optim_args.weight_decay
        self._nnpu = optim_args.nnpu
        self._combined_loss = optim_args.combined_loss

        self._optimizer = None
        self._write = optim_args.log_tensorboard

        if self._write:
            self._writer = SummaryWriter("tensorboard_logs/{}".format(seed))
        else:
            self._writer = None

    def initial_fit(self, model, data_wrapper, scaler):
        self._optimizer = create_optimizer(model.parameters(), self._optimizer_name,
                                           self._lr, self._momentum, self._nesterov, self._weight_decay)

        x_train, y_train = data_wrapper.get_init_train_data()
        x_val, y_val = data_wrapper.get_validation_data()

        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

        train_regular_nn(model, self._optimizer, F.cross_entropy, x_train, y_train, x_val, y_val,
                         self._epochs, self._early_stopping_iter, self._writer, "train_loss/0", self._write)

    def update_fit(self, model, data_wrapper, rate_tracker, scaler, update_num, *args):
        if not self._update:
            return model

        if not self._warm_start:
            model = self._model_fn(data_wrapper.dimension).to(model.device)
            self._optimizer = create_optimizer(model.parameters(), self._optimizer_name,
                                               self._lr, self._momentum, self._nesterov, self._weight_decay)

        x_temp, y_temp = data_wrapper.get_init_train_data()
        ignore_samples = len(x_temp)
        prior = np.sum(y_temp) / float(len(y_temp))
        prior = np.array(1 - prior)

        x_train, y_train = data_wrapper.get_all_data_for_model_fit_corrupt()
        x_val, y_val = data_wrapper.get_validation_data()

        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

        if self._combined_loss:
            train_loss_fn = PULossCombined(prior=prior, ignore_samples=ignore_samples, nnPU=self._nnpu)
        else:
            train_loss_fn = PULoss(prior=prior, nnPU=self._nnpu)

        val_loss_fn = PULoss(prior=prior, nnPU=self._nnpu)

        model = train_pu_nn(model, self._optimizer, train_loss_fn, val_loss_fn, x_train, y_train, x_val, y_val,
                                 self._epochs, self._early_stopping_iter, self._writer,
                                 "train_loss/{}".format(update_num), self._write)

        return model


def train_pu_nn(model, optimizer, train_loss_fn, val_loss_fn, x_train, y_train,
                x_val, y_val, epochs, early_stopping_iter, writer, writer_prefix, write=True):
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

    x_train = torch.from_numpy(x_train).float().to(model.device)
    y_train = torch.from_numpy(y_train).long().to(model.device)

    x_val = torch.from_numpy(x_val).float().to(model.device)
    y_val = torch.from_numpy(y_val).long().to(model.device)

    while not done:
        out = model(x_train)
        probs = F.softmax(out, dim=1)[:, 1]
        out = out[:, 1]

        train_loss = train_loss_fn(out, probs, y_train)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(train_loss.item())

        with torch.no_grad():
            # val_out = model(x_val)
            # val_loss = F.cross_entropy(val_out, y_val)
            val_out = model(x_val)
            val_probs = F.softmax(val_out, dim=1)[:, 1]
            val_out = val_out[:, 1]

            val_loss = val_loss_fn(val_out, val_probs, y_val)

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

    return model