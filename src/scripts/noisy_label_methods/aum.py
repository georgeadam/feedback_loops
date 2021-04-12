from typing import Dict

from datetime import datetime
import copy
import hydra
import os
import logging
import numpy as np
import pandas as pd
import torch
import torchvision
import logging
from tqdm import tqdm
import torch.nn.functional as F

from src.scripts.helpers.generic.train import load_data, eval_model, create_corrupted_labels, compute_loss
from src.utils.train import train_regular, train_lre
from src.utils.train.nn.utils import log_regular_losses

from src.models.lre import NN_LRE
from src.utils.data import StaticUpdateDataGenerator
from src.utils.optimizer import create_optimizer
from src.utils.save import CSV_FILE
from src.utils.rand import set_seed
from src.utils.str_formatting import SafeDict

from omegaconf import DictConfig
from settings import ROOT_DIR

logger = logging.getLogger(__name__)
os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs/aum.yaml")


def compute_margin(logits: torch.Tensor, labels: torch.Tensor):
    logits = logits.clone()
    label_logits = logits[torch.arange(len(labels)).to(labels.device), labels].clone()
    logits[torch.arange(len(labels)).to(labels.device), labels] = 0

    top_logits = torch.topk(logits, k=1, dim=1, largest=True).values.view(-1)

    return label_logits - top_logits


def train_aum(model, x_train, y_train, x_val, y_val, optimizer, epochs, early_stopping_iter):
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

        epoch += 1

    model.load_state_dict(best_params)
    logger.info("Best Train Loss: {} | Best Val Loss : {} at epoch: {}".format(best_train_loss.item(),
                                                                               best_val_loss.item(),
                                                                               best_epoch))

    if not done:
        logger.info("Stopped after: {} epochs, but could have kept improving loss.".format(epochs))

    return margins


def compute_aum_metrics(model, x, corrupted_targets, original_targets, margins):
    with torch.no_grad():
        logits = model(x)
        pred = torch.max(logits, 1)[1]

    pos_indices = (pred == 1).detach().cpu()

    margins = torch.stack(margins, dim=1)
    aum = torch.mean(margins, dim=1)
    aum = aum[-len(x):]
    corrupted_indices = np.where(original_targets != corrupted_targets)[0]
    num_corrupted = len(corrupted_indices)

    # Only care about positive samples with low AUM since the noise is asymmetric. Without indexing at the positive
    # indices only, we are underestimating the performance of AUM.
    sorted_indices = torch.argsort(aum).detach().cpu().numpy()
    sorted_indices = sorted_indices[pos_indices[sorted_indices]][:len(corrupted_indices)]

    intersection = len(np.intersect1d(sorted_indices, corrupted_indices))

    print("Out of {} corrupted samples, by taking the bottom {} sorted AUMs, we recover {}/{} corrupted samples".format(
        len(corrupted_indices), len(corrupted_indices), intersection, len(corrupted_indices)))

    if num_corrupted > 0:
        recall = intersection / float(num_corrupted)
    else:
        recall = 0.0

    return sorted_indices, aum, recall


def update_loop(data, data_args, model_args, optim_args, seed):
    set_seed(seed)
    model = NN_LRE(data["x_train"].shape[1], 2, model_args.hidden_layers, model_args.activation, model_args.device)
    optimizer = create_optimizer(model.params(), optim_args.optimizer, optim_args.lr,
                                 optim_args.momentum, optim_args.nesterov,
                                 optim_args.weight_decay)

    if optim_args.combine_train_val:
        x_val = data["x_val_reg"]
        y_val = data["y_val_reg"]

        x_cumulative = torch.cat([data["x_train"], data["x_val_lre"]])
        y_cumulative_corrupt = torch.cat([data["y_train"], data["y_val_lre"]])
        y_cumulative_clean = torch.cat([data["y_train"], data["y_val_lre"]])
    else:
        x_val = torch.cat([data["x_val_reg"], data["x_val_lre"]])
        y_val = torch.cat([data["y_val_reg"], data["y_val_lre"]])

        x_cumulative = data["x_train"]
        y_cumulative_corrupt = data["y_train"]
        y_cumulative_clean = data["y_train"]

    # Train initial model
    _ = train_regular(model, x_cumulative, y_cumulative_clean, x_val, y_val,
                      optimizer, optim_args.epochs, optim_args.early_stopping_iter,
                      None, None, False)
    initial_rates = eval_model(model, data["x_test"], data["y_test"])
    initial_rates["detection"] = 0.0

    x_update = copy.deepcopy(data["x_update"])
    y_update = copy.deepcopy(data["y_update"])

    update_wrapper = StaticUpdateDataGenerator(x_update, y_update, data_args.num_updates)
    cumulative_rates = {key: [initial_rates[key]] for key in initial_rates.keys()}

    for update_num, (x_update_partial, y_update_partial) in enumerate(update_wrapper, start=1):
        y_update_partial_corrupt = create_corrupted_labels(model, x_update_partial, y_update_partial)

        x_cumulative_temp = accumulate_data(x_cumulative, x_update_partial)
        y_cumulative_corrupt_temp = accumulate_data(y_cumulative_corrupt, y_update_partial_corrupt)

        # Compute margins by training a ghost model
        set_seed(seed)

        aum_model = NN_LRE(data["x_train"].shape[1], 2, model_args.hidden_layers, model_args.activation, model_args.device)

        optimizer = create_optimizer(aum_model.params(), optim_args.optimizer, optim_args.lr,
                                     optim_args.momentum, optim_args.nesterov,
                                     optim_args.weight_decay)
        margins = train_aum(aum_model, x_cumulative_temp, y_cumulative_corrupt_temp, x_val, y_val,
                            optimizer, optim_args.epochs, optim_args.early_stopping_iter)

        # We don't care about the margins on the training set, otherwise this will capture the label noise in the
        # training data that's not related to the feedback loop
        # Filter noisy samples based on margins
        bad_indices, _, detection_rate = compute_aum_metrics(aum_model, x_update_partial,
                                                             y_update_partial_corrupt.detach().cpu(), y_update_partial.detach().cpu(),
                                                             margins)
        good_indices = np.arange(len(x_update_partial))
        good_indices = np.setdiff1d(good_indices, bad_indices)
        x_cumulative = accumulate_data(x_cumulative, x_update_partial[good_indices])
        y_cumulative_corrupt = accumulate_data(y_cumulative_corrupt, y_update_partial_corrupt[good_indices])
        y_cumulative_clean = accumulate_data(y_cumulative_clean, y_update_partial[good_indices])

        # Train on clean samples
        set_seed(seed)

        if optim_args.from_scratch:
            model = NN_LRE(data["x_train"].shape[1], 2, model_args.hidden_layers, model_args.activation, model_args.device)

        optimizer = create_optimizer(model.params(), optim_args.optimizer, optim_args.lr,
                                     optim_args.momentum, optim_args.nesterov,
                                     optim_args.weight_decay)
        _ = train_regular(model, x_cumulative, y_cumulative_corrupt, x_val, y_val,
                          optimizer, optim_args.epochs, optim_args.early_stopping_iter,
                          None, None, False)

        rates = eval_model(model, data["x_test"], data["y_test"])
        rates["detection"] = detection_rate
        update_rates(cumulative_rates, rates)

    return cumulative_rates


def accumulate_data(data_cumulative, data_partial):
    if data_cumulative is None:
        data_cumulative = data_partial
    else:
        data_cumulative = torch.cat((data_cumulative, data_partial))

    return data_cumulative


def update_rates(cumulative_rates, rates):
    for key in rates.keys():
        if key not in cumulative_rates.keys():
            cumulative_rates[key] = [rates[key]]
        else:
            cumulative_rates[key].append(rates[key])


def loop(seed, args):
    set_seed(seed)
    data = load_data(args.data, args.model)

    rates = update_loop(data, args.data, args.model, args.optim, seed)
    rates["num_updates"] = list(range(len(rates["auc"])))
    rates["update_type"] = ["aum"] * len(rates["auc"])

    return rates


def collate_results(results):
    keys = results[0].keys()
    collated_results = {"seed": []}

    for i in range(len(results)):
        collated_results["seed"] += [i] * len(results[i][list(results[i].keys())[0]])
        for key in keys:
            if key not in collated_results.keys():
                collated_results[key] = results[i][key]
            else:
                collated_results[key] += results[i][key]

    return collated_results


@hydra.main(config_path=config_path)
def main(args: DictConfig):
    print(args.pretty())
    print("Saving to: {}".format(os.getcwd()))
    start_time = datetime.now()

    results = list(map(lambda x: loop(x, args), range(args.misc.seeds)))

    collated_results = collate_results(results)

    csv_file_name = CSV_FILE
    data = pd.DataFrame(collated_results)
    data.to_csv(csv_file_name, index=False, header=True)

    print("Script took: {} seconds".format(datetime.now() - start_time))


if __name__ == "__main__":
    main()