import copy
import logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils.str_formatting import SafeDict
from src.utils.optimizer import create_optimizer
from src.utils.train.nn.utils import compute_loss, log_regular_losses
from src.utils.train.nn.regular import train_regular_nn

logger = logging.getLogger(__name__)


class AUMNNTrainer:
    def __init__(self, model_fn, seed, warm_start, update, optimizer, lr, momentum,
                 nesterov, weight_decay, epochs, early_stopping_iter, device, **kwargs):
        self._warm_start = warm_start
        self._update = update
        self._epochs = epochs
        self._early_stopping_iter = early_stopping_iter
        self._model_fn = model_fn
        self._device = device

        self._optimizer_name = optimizer
        self._lr = lr
        self._momentum = momentum
        self._nesterov = nesterov
        self._weight_decay = weight_decay

        self._optimizer = None
        self._writer = SummaryWriter("tensorboard_logs/{}".format(seed))
        self._writer_prefix = "{type}/lre_clean/{update_num}/{name}"
        self._write = True

    def initial_fit(self, model, data_wrapper, scaler):
        self._optimizer = create_optimizer(model.parameters(), self._optimizer_name,
                                           self._lr, self._momentum, self._nesterov, self._weight_decay)

        x_train, y_train = data_wrapper.get_init_train_data()
        x_val, y_val = data_wrapper.get_validation_data()

        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

        train_regular_nn(model, self._optimizer, x_train, y_train, x_val, y_val,
                         self._epochs, self._early_stopping_iter, self._writer, "train_loss/0")

    def update_fit(self, model, data_wrapper, rate_tracker, scaler, update_num):
        if not self._warm_start:
            model = self._model_fn(data_wrapper.dimension).to(self._device)
            self._optimizer = create_optimizer(model.parameters(), self._optimizer_name,
                                               self._lr, self._momentum, self._nesterov, self._weight_decay)

        auxiliary_model = self._model_fn(data_wrapper.dimension).to(self._device)
        auxiliary_optimizer = create_optimizer(auxiliary_model.parameters(), self._optimizer_name, self._lr,
                                               self._momentum, self._nesterov, self._weight_decay)

        x_train, y_train = data_wrapper.get_all_data_for_model_fit()
        x_val, y_val = data_wrapper.get_validation_data()
        x_update, y_update = data_wrapper.get_current_update_batch()

        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)

        margins = train_aum(auxiliary_model, auxiliary_optimizer, x_train, y_train, x_val, y_val,
                            self._epochs, self._early_stopping_iter,
                            self._writer, self._writer_prefix.format_map(SafeDict(type="aum", update_num=update_num)), self._write)
        relevant_margins = margins[-len(x_update):]

        with torch.no_grad():
            auxiliary_out = auxiliary_model(scaler.transform(x_update))
            auxiliary_pred = torch.max(auxiliary_out, 1)[1].detach().cpu().numpy()

        fpr = rate_tracker.get_rates()["fpr"][-1]
        pos_indices = np.where(auxiliary_pred == 1)
        sorted_indices = torch.argsort(relevant_margins).detach().cpu().numpy()

        potentially_mislabeled_samples = int(fpr * np.sum(pos_indices))
        bad_indices = sorted_indices[pos_indices[sorted_indices]][:potentially_mislabeled_samples]
        good_indices = np.arange(len(x_update))
        good_indices = np.setdiff1d(good_indices, bad_indices)

        x_update, y_update = x_update[good_indices], y_update[good_indices]
        data_wrapper.store_current_update_batch(x_update, y_update)

        x_train, y_train = data_wrapper.get_all_data_for_model_fit()
        x_train = scaler.transform(x_train)

        model = train_regular_nn(model, self._optimizer, x_train, y_train, x_val, y_val,
                                 self._epochs, self._early_stopping_iter, self._writer,
                                 self._writer_prefix.format_map(SafeDict(type="regular", update_num=update_num)))

        return model


def train_aum(model, optimizer, x_train, y_train, x_val, y_val, epochs, early_stopping_iter, writer, writer_prefix,
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

    margins = []

    while not done:
        out = model(x_train)
        train_loss = F.cross_entropy(out, y_train)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(train_loss.item())

        with torch.no_grad():
            logits = out
            labels = y_train
            margin = compute_margin(logits, labels)
            margins.append(margin)

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

    return margins


def compute_margin(logits: torch.Tensor, labels: torch.Tensor):
    logits = logits.clone()
    label_logits = logits[torch.arange(len(labels)).to(labels.device), labels].clone()
    logits[torch.arange(len(labels)).to(labels.device), labels] = 0

    top_logits = torch.topk(logits, k=1, dim=1, largest=True).values.view(-1)

    return label_logits - top_logits