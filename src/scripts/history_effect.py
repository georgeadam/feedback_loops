import copy
import importlib
import numpy as np
import pandas as pd
import os
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

import importlib

from src.models.sklearn import lr
from src.utils.data import get_data_fn
from src.utils.metrics import eval_model
from src.utils.model import get_model_fn
from src.utils.update import update_model_feedback_with_training, update_model_feedback_with_training_cumulative
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp

from sklearn.model_selection import train_test_split

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="gaussian", choices=["gaussian", "sklearn", "mimic"], type=str)
parser.add_argument("--seeds", default=100, type=int)
parser.add_argument("--model", default="lr", type=str)

parser.add_argument("--n-train", default=1000, type=float)
parser.add_argument("--n-update", default=1000, type=float)
parser.add_argument("--n-test", default=5000, type=float)
parser.add_argument("--num-features", default=2, type=int)
parser.add_argument("--num-updates", default=100, type=int)

parser.add_argument("--m0", default=-1.0, type=float)
parser.add_argument("--m1", default=1.0, type=float)
parser.add_argument("--s0", default=1.0, type=float)
parser.add_argument("--s1", default=1.0, type=float)
parser.add_argument("--p0", default=0.5, type=float)
parser.add_argument("--p1", default=0.5, type=float)

parser.add_argument("--train-percentages", default=[0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    nargs="+")


def train_update_loop(model_fn, n_train, n_update, n_test, update_fn, num_updates, num_features, train_percentages, data_fn, seeds):
    seeds = np.arange(seeds)
    updated_results = {train_percentage: {"fpr": [], "tpr": [], "fnr": [], "tnr": []} for train_percentage in
                       train_percentages}

    for seed in seeds:
        print(seed)
        np.random.seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test, num_features=num_features)

        model = model_fn(num_features=num_features)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        initial_tnr, initial_fpr, initial_fnr, initial_tpr = eval_model(y_test, y_pred)

        for train_percentage in train_percentages:
            if train_percentage == 0.0:
                x_train_sub = np.empty((0, x_train.shape[1]), x_train.dtype)
                y_train_sub = np.empty((0), y_train.dtype)
            elif train_percentage == 1.0:
                x_train_sub = x_train
                y_train_sub = y_train
            else:
                x_train_sub, _, y_train_sub, __ = train_test_split(x_train, y_train, test_size=1 - train_percentage)

            new_model, rates = update_fn(model, x_train_sub, y_train_sub, x_update, y_update, x_test, y_test,
                                         num_updates, intermediate=True)

            updated_results[train_percentage]["fpr"].append([initial_fpr] + rates["fpr"])
            updated_results[train_percentage]["tpr"].append([initial_tpr] + rates["tpr"])
            updated_results[train_percentage]["fnr"].append([initial_fnr] + rates["fnr"])
            updated_results[train_percentage]["tnr"].append([initial_tnr] + rates["tnr"])

    return updated_results


def results_to_dataframe(results, train_percentages, seeds):
    data = {"type": [], "rate": [], "update": [], "train_percentage": []}

    for train_percentage in train_percentages:
        for key in results[train_percentage].keys():
            for i in range(seeds):
                data["type"] += [key] * (len(results[train_percentage][key][i]))
                data["rate"] += results[train_percentage][key][i]
                data["update"] += (np.arange(len(results[train_percentage][key][i]))).tolist()
                data["train_percentage"] += [train_percentage] * (len(results[train_percentage][key][i]))

    return pd.DataFrame(data)


def plot(data, num_updates, train_percentages, plot_path):
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)
    g = sns.lineplot(x="update", y="rate", hue="train_percentage",
                     data=data.loc[data["type"] == "fpr"], legend="full", ax=ax)

    ax.set_xlabel("Num Updates", size=30, labelpad=10.0)
    ax.set_ylabel("FPR", size=30, labelpad=10.0)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    ax.set_xlim([0, num_updates])

    legend = ax.legend(title="Train %", title_fontsize=24, labels=train_percentages, loc="upper left")
    legend.texts[0].set_size(20)

    fig.savefig("{}.{}".format(plot_path, "png"), bbox_inches='tight', dpi=600)
    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    results_dir = os.environ.get("HISTORY_EFFECT_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    model_fn = get_model_fn(args.model)

    data_fn = get_data_fn(args)
    results_non_cumulative = train_update_loop(model_fn, args.n_train, args.n_update, args.n_test, update_model_feedback_with_training,
                                               args.num_updates, args.num_features, args.train_percentages, data_fn, args.seeds)

    data_non_cumulative = results_to_dataframe(results_non_cumulative, args.train_percentages, args.seeds)

    plot_name = "update_fpr_non_cumulative_{}".format(args.data_type)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    plot(data_non_cumulative, args.num_updates, args.train_percentages, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)

    results_cumulative = train_update_loop(model_fn, args.n_train, args.n_update, args.n_test,
                                           update_model_feedback_with_training_cumulative,
                                           args.num_updates, args.num_features, args.train_percentages, data_fn,
                                           args.seeds)

    data_cumulative = results_to_dataframe(results_cumulative, args.train_percentages, args.seeds)

    plot_name = "update_fpr_cumulative_{}".format(args.data_type)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    plot(data_cumulative, args.num_updates, args.train_percentages, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)




if __name__ == "__main__":
    main(parser.parse_args())