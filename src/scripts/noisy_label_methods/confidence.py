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

from omegaconf import DictConfig
from settings import ROOT_DIR

logger = logging.getLogger(__name__)
os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs/confidence.yaml")


def update_loop(data, data_args, model_args, optim_args, seed, corrupt=False, drop=False):
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
        if corrupt:
            y_update_partial_corrupt = create_corrupted_labels(model, x_update_partial, y_update_partial)
        else:
            y_update_partial_corrupt = y_update_partial

        # Find noisy samples based on low confidence predictions
        if corrupt and drop:
            bad_indices, detection_rate = get_low_confidence_indices(model, x_update_partial,
                                                                     y_update_partial_corrupt, y_update_partial)
            good_indices = np.arange(len(x_update_partial))
            good_indices = np.setdiff1d(good_indices, bad_indices)
        else:
            good_indices = np.arange(len(x_update_partial))
            detection_rate = 0.0

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


def get_low_confidence_indices(model, x, y_corrupt, y_clean):
    with torch.no_grad():
        logits = model(x)
        pred = torch.max(logits, 1)[1]
        logits = logits[:, 1]

    pos_indices = (pred == 1).detach().cpu()
    corrupted_indices = np.where(y_clean.detach().cpu() != y_corrupt.detach().cpu())[0]
    num_corrupted = len(corrupted_indices)

    sorted_logit_indices = torch.argsort(logits.detach().cpu())
    sorted_logit_indices = sorted_logit_indices[pos_indices[sorted_logit_indices]][:len(corrupted_indices)]
    intersection = len(np.intersect1d(sorted_logit_indices, corrupted_indices))

    if num_corrupted > 0:
        recall = intersection / float(num_corrupted)
    else:
        recall = 0.0

    return sorted_logit_indices, recall


def update_rates(cumulative_rates, rates):
    for key in rates.keys():
        if key not in cumulative_rates.keys():
            cumulative_rates[key] = [rates[key]]
        else:
            cumulative_rates[key].append(rates[key])


def loop(seed, args):
    set_seed(seed)
    data = load_data(args.data, args.model)
    results = {"train_type": [], "num_updates": []}

    feedback_rates_drop = update_loop(data, args.data, args.model, args.optim, seed, corrupt=True, drop=True)

    grouped_rates = {"confidence_drop_feedback": feedback_rates_drop}

    for train_type in grouped_rates.keys():
        for rate_name in grouped_rates[train_type].keys():
            if rate_name not in results.keys():
                results[rate_name] = grouped_rates[train_type][rate_name]
            else:
                results[rate_name] += grouped_rates[train_type][rate_name]

        results["num_updates"] += list(range(len(grouped_rates[train_type][rate_name])))
        results["train_type"] += [train_type] * len(grouped_rates[train_type][rate_name])

    return results


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