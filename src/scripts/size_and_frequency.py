import copy
import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import importlib

from src.utils.data import get_data_fn
from src.utils.metrics import eval_model
from src.utils.model import get_model_fn
from src.utils.update import get_update_fn
from src.utils.rand import set_seed
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp


from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="gaussian", choices=["gaussian", "sklearn", "mimic"], type=str)
parser.add_argument("--seeds", default=1, type=int)
parser.add_argument("--model", default="lr_pytorch", type=str)

parser.add_argument("--n-train", default=1000, type=float)
parser.add_argument("--n-test", default=50000, type=float)
parser.add_argument("--num-features", default=2, type=int)
parser.add_argument("--num-updates", default=500, type=int)

parser.add_argument("--m0", default=-1.0, type=float)
parser.add_argument("--m1", default=1.0, type=float)
parser.add_argument("--s0", default=1.0, type=float)
parser.add_argument("--s1", default=1.0, type=float)
parser.add_argument("--p0", default=0.5, type=float)
parser.add_argument("--p1", default=0.5, type=float)

parser.add_argument("--sizes", default=[500, 1000, 2500, 5000, 10000], nargs="+")
parser.add_argument("--noise", default=0.0, type=float)

parser.add_argument("--lr", default=1.0, type=float)
parser.add_argument("--iterations", default=1000, type=int)
parser.add_argument("--importance", default=1.0, type=float)

parser.add_argument("--update-type", default="feedback_confidence", type=str)


def train_update_loop(model_fn, n_train, n_test, sizes, num_features, updates, data_fn, update_fn, noise, seeds):
    seeds = np.arange(seeds)
    results = {size: {"fpr": [], "fnr": [], "tpr": [], "tnr": []} for size in sizes}

    for seed in seeds:
        for size in sizes:
            set_seed(seed)

            x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, size, n_test, num_features=num_features,
                                                                           noise=noise)

            model = model_fn(num_features=num_features)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            initial_tnr, initial_fpr, initial_fnr, initial_tpr = eval_model(y_test, y_pred)

            new_model, rates = update_fn(model, x_train, y_train, x_update, y_update, x_test, y_test, updates, intermediate=True)

            results[size]["fpr"].append([initial_fpr] + rates["fpr"])
            results[size]["fnr"].append([initial_fnr] + rates["fnr"])
            results[size]["tpr"].append([initial_tpr] + rates["tpr"])
            results[size]["tnr"].append([initial_tnr] + rates["tnr"])

    return results


def gold_standard_loop(model_fn, n_train, n_update, n_test, num_features, data_fn, noise, seeds):
    seeds = np.arange(seeds)
    results = {"fpr": [], "fnr": [], "tpr": [], "tnr": []}

    for seed in seeds:
        np.random.seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       noise=noise, num_features=num_features)

        model = model_fn(num_features=num_features)
        model.fit(np.concatenate((x_train, x_update)), np.concatenate((y_train, y_update)))

        y_pred = model.predict(x_test)
        gold_standard_tnr, gold_standard_fpr, gold_standard_fnr, gold_standard_tpr = eval_model(y_test, y_pred)

        results["fpr"].append(gold_standard_fpr)
        results["fnr"].append(gold_standard_fnr)
        results["tpr"].append(gold_standard_tpr)
        results["tnr"].append(gold_standard_tnr)

    return results


def results_to_dataframe(results, sizes, updates, seeds):
    data = {"num_updates": [], "Total Update\n Samples": [], "rate": [], "type": []}

    names = ["fpr", "fnr"]

    for size in sizes:
        for name in names:
            for i in range(seeds):
                data["rate"] += results[size][name][i]
                data["type"] += [name] * len(results[size][name][i])
                data["Total Update\n Samples"] += [size] * len(results[size][name][i])
                data["num_updates"] += (np.arange(updates + 1)).tolist()

    data = pd.DataFrame(data)

    return data


def plot(data, sizes, updates, gs, plot_path):
    fig = plt.figure(figsize=(13,9))
    # fig.suptitle("FPR vs Num Updates (fixed batch size)", fontsize=24)

    ax = fig.add_subplot(111)

    g = sns.lineplot(x="num_updates", y="rate", hue="Total Update\n Samples", style="type", legend="full", data=data.loc[data["type"] == "fpr"], ax=ax, palette="bright")

    ax.set_ylabel("FPR", fontsize=30, labelpad=10.0)
    ax.set_xlabel("Num Updates", fontsize=30, labelpad=10.0)

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    ax.set_xlim([0, updates])


    ax.axhline(gs, ls="--")

    legend = ax.legend(title="Samples Per\nUpdate", title_fontsize=30, labels=["{}".format(size / updates) for size in sizes], loc="upper left")
    legend.texts[0].set_size(20)

    fig.show()

    fig.savefig("{}.{}".format(plot_path, "png"), bbox_inches='tight', dpi=600)
    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    results_dir = os.environ.get("SIZE_AND_FREQUENCY_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    model_fn = get_model_fn(args)
    data_fn = get_data_fn(args)
    update_fn = get_update_fn(args)

    gold_standard_fprs = gold_standard_loop(model_fn, args.n_train, args.sizes[-1], args.n_test, args.num_features, data_fn,
                                            args.noise, args.seeds)
    results = train_update_loop(model_fn, args.n_train, args.n_test, args.sizes, args.num_features, args.num_updates,
                                data_fn, update_fn, args.noise, args.seeds)

    data = results_to_dataframe(results, args.sizes, args.num_updates, args.seeds)
    gs = np.mean(gold_standard_fprs["fpr"])

    plot_name = "fpr_size_frequency_lineplot_{}".format(args.data_type)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    plot(data, args.sizes, args.num_updates, gs, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)


if __name__ == "__main__":
    main(parser.parse_args())