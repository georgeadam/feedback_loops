import copy
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


import hydra.experimental
import os
import sys
sys.path.append("..")

from datetime import datetime
from omegaconf import DictConfig
from settings import ROOT_DIR

from sklearn.model_selection import train_test_split

from src.utils.metrics import compute_all_rates
from src.utils.preprocess import get_scaler
from src.utils.rand import set_seed
from src.utils.data import get_data_fn, UpdateDataWrapper
from src.utils.save import CSV_FILE

from src.models.lre import NN_Meta
from src.scripts.helpers.lre.lre import load_data, train_regular, train_lre, eval_model, create_corrupted_labels


os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs/lre_multistep.yaml")


def update_loop(x_train, y_train, x_val, y_val, x_update, y_update, x_test, y_test,
                args, train_fn, seed, corrupt=False):
    set_seed(seed)
    model = NN_Meta(x_train.shape[1], 2, args.model.hidden_layers, args.model.activation, args.model.device)
    optimizer = torch.optim.SGD(model.params(), args.optim.lr)

    _ = train_regular(model, x_train, y_train, optimizer, args.optim.epochs, args.optim.regular_tol)
    initial_rates = eval_model(model, x_test, y_test)

    x_update = copy.deepcopy(x_update)
    y_update = copy.deepcopy(y_update)

    update_wrapper = UpdateDataWrapper(x_update, y_update, args.data.num_updates)
    x_cumulative = x_train
    y_cumulative = y_train

    cumulative_rates = {key: [initial_rates[key]] for key in initial_rates.keys()}

    for x_update_partial, y_update_partial in update_wrapper:
        if corrupt:
            y_update_partial = create_corrupted_labels(model, x_update_partial, y_update_partial)

        x_cumulative, y_cumulative = accumulate_data(x_cumulative, y_cumulative,
                                                                   x_update_partial, y_update_partial)

        set_seed(seed)
        model = NN_Meta(x_train.shape[1], 2, args.model.hidden_layers, args.model.activation, args.model.device)
        optimizer = torch.optim.SGD(model.params(), args.optim.lr, momentum=args.optim.momentum,
                                    weight_decay=args.optim.weight_decay)

        if train_fn is train_regular:
            _ = train_regular(model, x_cumulative, y_cumulative, optimizer, args.optim.epochs, args.optim.regular_tol)
        elif train_fn is train_lre:
            _, _, _ = train_lre(model, x_cumulative, y_cumulative, x_val, y_val, optimizer, args,
                                args.model.device)

        rates = eval_model(model, x_test, y_test)
        update_rates(cumulative_rates, rates)

    return cumulative_rates


def accumulate_data(x_update_cumulative, y_update_cumulative, x_update_partial, y_update_partial):
    if x_update_cumulative is None:
        x_update_cumulative = x_update_partial
        y_update_cumulative = y_update_partial
    else:
        x_update_cumulative = torch.cat((x_update_cumulative, x_update_partial))
        y_update_cumulative = torch.cat((y_update_cumulative, y_update_partial))

    return x_update_cumulative, y_update_cumulative


def update_rates(cumulative_rates, rates):
    for key in rates.keys():
        if key not in cumulative_rates.keys():
            cumulative_rates[key] = [rates[key]]
        else:
            cumulative_rates[key].append(rates[key])


def loop(seed, args):
    results = {"train_type": [], "num_updates": []}
    set_seed(seed)
    (x_train, y_train, x_val, y_val,
     x_update, y_update, x_test, y_test) = load_data(args.data, args.model)

    regular_clean_rates = update_loop(x_train, y_train, x_val, y_val, x_update, y_update,
                                      x_test, y_test, args, train_regular, seed, corrupt=False)

    regular_corrupt_rates = update_loop(x_train, y_train, x_val, y_val, x_update, y_update,
                                        x_test, y_test, args, train_regular, seed, corrupt=True)

    lre_clean_rates = update_loop(x_train, y_train, x_val, y_val, x_update, y_update,
                                  x_test, y_test, args, train_lre, seed, corrupt=False)

    lre_corrupt_rates = update_loop(x_train, y_train, x_val, y_val, x_update, y_update,
                                    x_test, y_test, args, train_lre, seed, corrupt=True)

    grouped_rates = {"regular_fused_clean": regular_clean_rates, "regular_fused_corrupt": regular_corrupt_rates,
                     "lre_fused_clean": lre_clean_rates, "lre_fused_corrupt": lre_corrupt_rates}

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
    collated_results = {}

    for i in range(len(results)):
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

    pool = mp.Pool(processes=args.misc.seeds)
    results = list(map(lambda x: loop(x, args), range(args.misc.seeds)))
    # out = [p.get() for p in results]
    collated_results = collate_results(results)

    csv_file_name = CSV_FILE
    data = pd.DataFrame(collated_results)
    data.to_csv(csv_file_name, index=False, header=True)

    print("Script took: {} seconds".format(datetime.now() - start_time))


if __name__ == "__main__":
    main()