import logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils.str_formatting import SafeDict
from src.utils.optimizer import create_optimizer
from src.utils.train.nn.regular import train_regular_nn

logger = logging.getLogger(__name__)


class MonteCarloShapleyNNTrainer:
    def __init__(self, model_fn, seed, warm_start, update, regular_optim_args, shapley_optim_args, **kwargs):
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

        self._epochs_shapley = shapley_optim_args.epochs
        self._early_stopping_iter_shapley = shapley_optim_args.early_stopping_iter
        self._optimizer_name_shapley = shapley_optim_args.optimizer
        self._lr_shapley = shapley_optim_args.lr
        self._momentum_shapley = shapley_optim_args.momentum
        self._nesterov_shapley = shapley_optim_args.nesterov
        self._weight_decay_shapley = shapley_optim_args.weight_decay
        self._shapley_runs = shapley_optim_args.runs

        self._optimizer = None
        self._writer_prefix = "{type}/{update_num}/{name}"
        self._write = regular_optim_args.log_tensorboard

        if self._write:
            self._writer = SummaryWriter("tensorboard_logs/{}".format(seed))
        else:
            self._writer = None

    def initial_fit(self, model, data_wrapper, scaler):
        self._optimizer = create_optimizer(model.parameters(), self._optimizer_name_regular,
                                           self._lr_regular, self._momentum_regular,
                                           self._nesterov_regular, self._weight_decay_regular)

        x_train, y_train = data_wrapper.get_init_train_data()
        x_val_reg, y_val_reg = data_wrapper.get_val_data_regular()

        x_train, x_val_reg = scaler.transform(x_train), scaler.transform(x_val_reg)

        train_regular_nn(model, self._optimizer, F.cross_entropy, x_train, y_train, x_val_reg, y_val_reg,
                         self._epochs_regular, self._early_stopping_iter_regular, self._writer,
                         self._writer_prefix.format_map(SafeDict(type="regular", update_num=0)), self._write)

    def update_fit(self, model, data_wrapper, rate_tracker, scaler, update_num, *args):
        if not self._update:
            return model

        if not self._warm_start:
            new_model = self._model_fn(data_wrapper.dimension).to(model.device)
            self._optimizer = create_optimizer(new_model.parameters(), self._optimizer_name_regular,
                                               self._lr_regular, self._momentum_regular, self._nesterov_regular,
                                               self._weight_decay_regular)
        else:
            new_model = model

        x_train, y_train = data_wrapper.get_all_data_for_model_fit_corrupt()
        x_val_reg, y_val_reg = data_wrapper.get_val_data_regular()
        x_val_shapley, y_val_shapley = data_wrapper.get_val_data_shapley()
        x_update, y_update = data_wrapper.get_current_update_batch_corrupt()

        x_train, x_val_reg, x_val_shapley = scaler.transform(x_train), scaler.transform(x_val_reg), scaler.transform(x_val_shapley)
        eval_fn = test_performance(x_val_shapley, y_val_shapley, model.device)
        optimizer_fn = optimizer_wrapper(self._optimizer_name_shapley, self._lr_shapley, self._momentum_shapley,
                                         self._nesterov_shapley, self._weight_decay_shapley)

        values, _ = mc_shapley(self._model_fn, optimizer_fn, x_train, y_train, x_val_reg, y_val_reg, eval_fn, self._epochs_shapley,
                              self._early_stopping_iter_shapley, self._shapley_runs, model.device)
        values = values[-len(x_update):]

        with torch.no_grad():
            out = model(scaler.transform(x_update))
            pred = torch.max(out, 1)[1].detach().cpu().numpy()

        fpr = rate_tracker.get_rates()["fpr"][-1]
        pos_indices = (pred == 1)
        sorted_indices = torch.argsort(values).detach().cpu().numpy()

        potentially_mislabeled_samples = int(fpr * np.sum(pos_indices))
        bad_indices = sorted_indices[pos_indices[sorted_indices]][:potentially_mislabeled_samples]
        good_indices = np.arange(len(x_update))
        good_indices = np.setdiff1d(good_indices, bad_indices)

        x_update, y_update = x_update[good_indices], y_update[good_indices]
        data_wrapper.store_current_update_batch_corrupt(x_update, y_update)

        x_train, y_train = data_wrapper.get_all_data_for_model_fit_corrupt()
        x_train = scaler.transform(x_train)

        new_model = train_regular_nn(new_model, self._optimizer, F.cross_entropy, x_train, y_train, x_val_reg, y_val_reg,
                                     self._epochs_regular, self._early_stopping_iter_regular, self._writer,
                                     self._writer_prefix.format_map(SafeDict(type="regular", update_num=update_num)), self._write)

        return new_model


def test_performance(x, y, device):
    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).long().to(device)

    def inner(model):
        model.eval()
        out = model(x)
        return - F.cross_entropy(out, y)

    return inner


def optimizer_wrapper(name, lr, momentum, nesterov, weight_decay):
    def inner(params):
        return create_optimizer(params, name, lr, momentum, nesterov, weight_decay)

    return inner


def mc_shapley(model_fn, optimizer_fn, x_train, y_train, x_val, y_val, eval_fn, epochs, early_stopping_iter, runs, device):
    values = torch.zeros(len(x_train))
    history = []
    t = 0

    # x_train = torch.from_numpy(x_train).float().to(device)
    # y_train = torch.from_numpy(y_train).long().to(device)
    #
    # x_val = torch.from_numpy(x_val).float().to(device)
    # y_val = torch.from_numpy(y_val).long().to(device)

    while t < runs:
        t += 1
        permutation = torch.randperm(len(x_train)).long()
        model = model_fn(x_train.shape[1]).to(device)
        initial_value = eval_fn(model)
        prev_value = initial_value

        cumulative_j = []
        print("On run: {}".format(t))

        for j in permutation:
            cumulative_j.append(j)
            model = model_fn(x_train.shape[1]).to(device)
            optimizer = optimizer_fn(model.parameters())

            train_regular_nn(model, optimizer, F.cross_entropy, x_train[cumulative_j].reshape(len(cumulative_j), -1), y_train[cumulative_j],
                             x_val, y_val, epochs, early_stopping_iter, None, None, False)

            with torch.no_grad():
                cur_value = eval_fn(model)

            values[j] = ((t - 1) / t) * values[j] + (1 / t) * (cur_value - prev_value)
            prev_value = cur_value

        history.append(values.clone().detach())

    return values, history
