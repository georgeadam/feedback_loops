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
config_path = os.path.join(ROOT_DIR, "configs/data_shapley.yaml")


def create_model(num_features, classes, hidden_layers, activation, device):
    def inner():
        return NN_LRE(num_features, classes, hidden_layers, activation, device)

    return inner


def test_performance(x, y):
    def inner(model):
        model.eval()
        out = model(x)
        return - F.cross_entropy(out, y)

    return inner


def train_step(model, optimizer, x, y):
    model.train()
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def gradient_shapley(model_fn, x, y, eval_test, train_fn, epochs):
    values = torch.zeros(len(x))
    t = 0

    while t < epochs:
        t += 1
        permutation = torch.randperm(len(x)).long()
        model = model_fn().to(x.device)
        optimizer = torch.optim.SGD(model.params(), lr=0.01)
        initial_value = eval_test(model)
        prev_value = initial_value

        for j in permutation:
            train_fn(model, optimizer, x[j].view(1, -1), y[j].view(1))
            with torch.no_grad():
                cur_value = eval_test(model)

            values[j] = ((t - 1) / t) * values[j] + (1 / t) * (cur_value - prev_value)
            prev_value = cur_value

    return values


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
        model = NN_LRE(data["x_train"].shape[1], 2, model_args.hidden_layers, model_args.activation, model_args.device)
        optimizer = create_optimizer(model.params(), optim_args.optimizer, optim_args.lr,
                                     optim_args.momentum, optim_args.nesterov,
                                     optim_args.weight_decay)
        model_fn = create_model(data["x_train"].shape[1], 2, model_args.hidden_layers, model_args.activation, model_args.device)
        eval_test = test_performance(data["x_test"], data["y_test"])
        train_fn = train_step

        values = gradient_shapley(model_fn, x_cumulative_temp, y_cumulative_corrupt_temp, eval_test, train_fn, optim_args.shapley_runs)

        # We don't care about the margins on the training set, otherwise this will capture the label noise in the
        # training data that's not related to the feedback loop
        # Filter noisy samples based on margins

        bad_indices, detection_rate = get_low_value_indices(model, values, x_update_partial,
                                                            y_update_partial_corrupt.detach().cpu(), y_update_partial.detach().cpu())
        good_indices = np.arange(len(x_update_partial))
        good_indices = np.setdiff1d(good_indices, bad_indices)

        x_cumulative = accumulate_data(x_cumulative, x_update_partial[good_indices])
        y_cumulative_corrupt = accumulate_data(y_cumulative_corrupt, y_update_partial_corrupt[good_indices])
        y_cumulative_clean = accumulate_data(y_cumulative_clean, y_update_partial[good_indices])

        # Train on clean samples
        set_seed(seed)
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


def get_low_value_indices(model, values, x, y_corrupt, y_clean):
    with torch.no_grad():
        logits = model(x)
        pred = torch.max(logits, 1)[1]

    pos_indices = (pred == 1).detach().cpu()

    values = values[-len(x):]

    corrupted_indices = np.where(y_clean != y_corrupt)[0]
    num_corrupted = len(corrupted_indices)

    # Only care about positive samples with low AUM since the noise is asymmetric. Without indexing at the positive
    # indices only, we are underestimating the performance of AUM.
    sorted_indices = torch.argsort(values).detach().cpu().numpy()
    sorted_indices = sorted_indices[pos_indices[sorted_indices]][:len(corrupted_indices)]

    intersection = len(np.intersect1d(sorted_indices, corrupted_indices))

    print("Out of {} corrupted samples, by taking the bottom {} sorted AUMs, we recover {}/{} corrupted samples".format(
        len(corrupted_indices), len(corrupted_indices), intersection, len(corrupted_indices)))

    if num_corrupted > 0:
        recall = intersection / float(num_corrupted)
    else:
        recall = 0.0

    return sorted_indices, recall


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
    rates["update_type"] = ["data_shapley"] * len(rates["auc"])

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