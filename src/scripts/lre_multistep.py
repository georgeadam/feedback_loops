import copy
import multiprocessing as mp
import pandas as pd
import torch

from torch.utils.tensorboard import SummaryWriter


import hydra.experimental
import os
import sys
sys.path.append("..")

from datetime import datetime
from omegaconf import DictConfig
from settings import ROOT_DIR

from src.utils.optimizer import create_optimizer
from src.utils.rand import set_seed
from src.utils.data import UpdateDataWrapper
from src.utils.save import CSV_FILE
from src.utils.str_formatting import SafeDict

from src.models.lre import NN_Meta
from src.scripts.helpers.generic.train import load_data, train_regular, train_lre, eval_model, create_corrupted_labels


os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs/lre_multistep.yaml")
# config_path = os.path.join(ROOT_DIR, "configs")
# config_name = "temp"


def update_loop(data, data_args, model_args, optim_args, train_fn, seed, writer, writer_prefix, corrupt=False):
    set_seed(seed)
    model = NN_Meta(data["x_train"].shape[1], 2, model_args.hidden_layers, model_args.activation, model_args.device)
    optimizer = create_optimizer(model.params(), optim_args.regular.optimizer, optim_args.regular.lr,
                                 optim_args.regular.momentum, optim_args.regular.weight_decay)

    _ = train_regular(model, data["x_train"], data["y_train"], data["x_val_reg"], data["y_val_reg"],
                      optimizer, optim_args.regular.epochs, optim_args.regular.early_stopping_iter,
                      writer, writer_prefix.format_map(SafeDict(update_num="0", type="initial")))
    initial_rates = eval_model(model, data["x_test"], data["y_test"])

    x_update = copy.deepcopy(data["x_update"])
    y_update = copy.deepcopy(data["y_update"])

    update_wrapper = UpdateDataWrapper(x_update, y_update, data_args.num_updates)
    x_cumulative = data["x_train"]
    y_cumulative = data["y_train"]

    cumulative_rates = {key: [initial_rates[key]] for key in initial_rates.keys()}

    for update_num, (x_update_partial, y_update_partial) in enumerate(update_wrapper, start=1):
        if corrupt:
            y_update_partial = create_corrupted_labels(model, x_update_partial, y_update_partial)

        x_cumulative, y_cumulative = accumulate_data(x_cumulative, y_cumulative,
                                                                   x_update_partial, y_update_partial)

        set_seed(seed)
        model = NN_Meta(data["x_train"].shape[1], 2, model_args.hidden_layers, model_args.activation, model_args.device)

        if train_fn is train_regular:
            optimizer = create_optimizer(model.params(), optim_args.regular.optimizer, optim_args.regular.lr,
                                         optim_args.regular.momentum, optim_args.regular.weight_decay)
            _ = train_regular(model, x_cumulative, y_cumulative, data["x_val_reg"], data["y_val_reg"],
                              optimizer, optim_args.regular.epochs, optim_args.regular.early_stopping_iter,
                              writer, writer_prefix.format_map(SafeDict(update_num=str(update_num), type="update")))
        elif train_fn is train_lre:
            optimizer = create_optimizer(model.params(), optim_args.lre.optimizer, optim_args.lre.lr,
                                         optim_args.lre.momentum, optim_args.lre.weight_decay)
            _, _, _ = train_lre(model, x_cumulative, y_cumulative, data["x_val_reg"], data["y_val_reg"],
                                data["x_val_lre"], data["y_val_lre"], optimizer, model_args, optim_args.lre,
                                writer, writer_prefix.format_map(SafeDict(update_num=str(update_num), type="update")), model_args.device)

        rates = eval_model(model, data["x_test"], data["y_test"])
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
    writer = SummaryWriter("tensorboard_logs/{}".format(seed))
    results = {"train_type": [], "num_updates": []}
    set_seed(seed)
    data = load_data(args.data, args.model)

    writer_prefix = "{type}/regular_clean/{update_num}/{name}"
    regular_clean_rates = update_loop(data, args.data, args.model, args.optim, train_regular, seed, writer, writer_prefix, corrupt=False)

    writer_prefix = "{type}/regular_corrupt/{update_num}/{name}"
    regular_corrupt_rates = update_loop(data, args.data, args.model, args.optim, train_regular, seed, writer, writer_prefix, corrupt=True)

    writer_prefix = "{type}/lre_clean/{update_num}/{name}"
    lre_clean_rates = update_loop(data, args.data, args.model, args.optim, train_lre, seed, writer, writer_prefix, corrupt=False)

    writer_prefix = "{type}/lre_corrupt/{update_num}/{name}"
    lre_corrupt_rates = update_loop(data, args.data, args.model, args.optim, train_lre, seed, writer, writer_prefix, corrupt=True)

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