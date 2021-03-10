import copy
import logging

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils.str_formatting import SafeDict
from src.utils.optimizer import create_optimizer
from src.utils.train.nn.utils import compute_loss
from src.utils.train.nn.regular import train_regular_nn

logger = logging.getLogger(__name__)


class LRENNTrainer:
    def __init__(self, model_fn, seed, warm_start, update, regular_optim_args, lre_optim_args, **kwargs):
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

        self._epochs_lre = lre_optim_args.epochs
        self._early_stopping_iter_lre = lre_optim_args.early_stopping_iter
        self._optimizer_name_lre = lre_optim_args.optimizer
        self._lr_lre = lre_optim_args.lr
        self._momentum_lre = lre_optim_args.momentum
        self._nesterov_lre = lre_optim_args.nesterov
        self._weight_decay_lre = lre_optim_args.weight_decay

        self._optimizer_lre = None
        self._writer = SummaryWriter("tensorboard_logs/{}".format(seed))
        self._writer_prefix = "{type}/{update_num}/{name}"
        self._write = True

    def initial_fit(self, model, data_wrapper, scaler):
        optimizer = create_optimizer(model.params(), self._optimizer_name_regular,
                                           self._lr_regular, self._momentum_regular,
                                           self._nesterov_regular, self._weight_decay_regular)

        x_train, y_train = data_wrapper.get_init_train_data()
        x_val, y_val = data_wrapper.get_validation_data()

        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

        train_regular_nn(model, optimizer, x_train, y_train, x_val, y_val,
                         self._epochs_regular, self._early_stopping_iter_regular, self._writer,
                         self._writer_prefix.format_map(SafeDict(type="regular", update_num=0)))

    def update_fit(self, model, data_wrapper, rate_tracker, scaler, update_num):
        if not self._warm_start:
            model = self._model_fn(data_wrapper.dimension).to(model.device)
            self._optimizer_lre = create_optimizer(model.params(), self._optimizer_name_lre,
                                               self._lr_lre, self._momentum_lre,
                                                   self._nesterov_lre, self._weight_decay_lre)
        elif self._optimizer_lre is None:
            self._optimizer_lre = create_optimizer(model.params(), self._optimizer_name_lre,
                                                   self._lr_lre, self._momentum_lre,
                                                   self._nesterov_lre, self._weight_decay_lre)

        x_train, y_train = data_wrapper.get_all_data_for_model_fit()
        x_val_lre, y_val_lre = data_wrapper.get_val_data_lre()
        x_val_reg, y_val_reg = data_wrapper.get_val_data_regular()

        x_train, x_val_lre, x_val_regular = scaler.transform(x_train), \
                                            scaler.transform(x_val_lre), \
                                            scaler.transform(x_val_reg)

        train_lre(model, self._model_fn, x_train, y_train, x_val_reg, y_val_reg, x_val_lre, y_val_lre, self._optimizer_lre,
                  self._lr_lre, self._epochs_lre, self._early_stopping_iter_lre,
                  self._writer, self._writer_prefix.format_map(SafeDict(type="lre", update_num=update_num)))

        return model


def train_lre(model, model_fn, x_train, y_train, x_val_reg, y_val_reg, x_val_lre, y_val_lre, optimizer,
              lr, epochs, early_stopping_iter, writer, writer_prefix):
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

    x_train = torch.from_numpy(x_train).float().to(model.device)
    y_train = torch.from_numpy(y_train).long().to(model.device)

    x_val_lre = torch.from_numpy(x_val_lre).float().to(model.device)
    y_val_lre = torch.from_numpy(y_val_lre).long().to(model.device)

    x_val_reg = torch.from_numpy(x_val_reg).float().to(model.device)
    y_val_reg = torch.from_numpy(y_val_reg).long().to(model.device)

    while not done:
        model.train()
        # Line 2 get batch of data
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_model = model_fn(x_train.shape[1]).to(model.device)
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
        meta_model.update_params(lr, source_params=grads)

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

        if no_train_improvement > early_stopping_iter:
            done = True
            logger.info("No improvement in train loss for {} epochs at epoch: {}. Stopping.".format(early_stopping_iter,
                                                                                                    epoch))

        if no_val_improvement > early_stopping_iter:
            done = True
            logger.info("No improvement in validation loss for {} epochs at epoch: {}. Stopping.".format(early_stopping_iter,
                                                                                                         epoch))

        if epoch % 100 == 0:
            logger.info("Epoch: {} | Weighted Train Loss: {} | Unweighted Train Loss: {} | LRE Val Loss: {} | Reg Val Loss: {}".format(epoch,
                                                                                                                                       l_f.item(),
                                                                                                                                       unweighted_loss.item(),
                                                                                                                                       l_g_meta.item(),
                                                                                                                                       val_loss.item()))

        if epoch > epochs:
            break

        log_lre_losses(writer, writer_prefix, unweighted_loss.item(), val_loss.item(), l_g_meta.item(), epoch)

        epoch += 1

    model.load_state_dict(best_params)
    logger.info("Best Train Loss: {} | Best LRE Val Loss: {} | Best Reg Val Loss Achieved: {} at epoch: {}".format(best_train_loss.item(),
                                                                                                                   best_val_loss.item(),
                                                                                                                   l_g_meta.item(),
                                                                                                                   best_epoch))

    if not done:
        logger.info("Stopped after: {} epochs, but could have kept improving loss.".format(epochs))

    return meta_losses_clean, net_losses, w


def log_lre_losses(writer, writer_prefix, train_loss, val_loss_reg, val_loss_lre, epoch):
    writer.add_scalar(writer_prefix.format(name="train_loss"), train_loss, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss_reg"), val_loss_reg, epoch)
    writer.add_scalar(writer_prefix.format(name="val_loss_lre"), val_loss_lre, epoch)