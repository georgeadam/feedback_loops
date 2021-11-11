import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.optimizer import create_optimizer
from src.utils.train import train_regular_nn

logger = logging.getLogger(__name__)


class BalancedReweightNNTrainer:
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

        pos = float(len(y_train)) / (2 * np.sum(y_train))
        neg = float(len(y_train)) / (2 * (len(y_train) - np.sum(y_train)))

        train_regular_nn(model, self._optimizer,
                         torch.nn.CrossEntropyLoss(weight=torch.tensor([neg, pos]).to(model.device).float()),
                         x_train, y_train, x_val, y_val, self._epochs, self._early_stopping_iter, self._writer,
                         "train_loss/0", self._write)

    def update_fit(self, model, data_wrapper, rate_tracker, scaler, update_num, *args):
        if not self._update:
            return model

        if not self._warm_start:
            model = self._model_fn(data_wrapper.dimension).to(model.device)
            self._optimizer = create_optimizer(model.parameters(), self._optimizer_name,
                                               self._lr, self._momentum, self._nesterov, self._weight_decay)

        x_train, y_train = data_wrapper.get_all_data_for_model_fit_corrupt()
        x_val, y_val = data_wrapper.get_validation_data()
        _, y_temp = data_wrapper.get_init_train_data()

        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

        pos = float(len(y_temp)) / (2 * np.sum(y_temp))
        neg = float(len(y_temp)) / (2 * (len(y_temp) - np.sum(y_temp)))

        model = train_regular_nn(model, self._optimizer,
                                 torch.nn.CrossEntropyLoss(weight=torch.tensor([neg, pos]).to(model.device).float()),
                                 x_train, y_train, x_val, y_val, self._epochs, self._early_stopping_iter, self._writer,
                                 "train_loss/{}".format(update_num), self._write)

        return model