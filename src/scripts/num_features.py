import copy
import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

import importlib

from src.models.sklearn import lr
from src.utils.data import get_data_fn
from src.utils.metrics import eval_model
from src.utils.model import get_model_fn
from src.utils.update import update_model_feedback
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp


from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="gaussian", choices=["gaussian", "sklearn"], type=str)
parser.add_argument("--seeds", default=100, type=int)
parser.add_argument("--model", default="lr", type=str)

parser.add_argument("--n-train", default=10000, type=int)
parser.add_argument("--n-update", default=10000, type=int)
parser.add_argument("--n-test", default=10000, type=int)
parser.add_argument("--features", default=10, type=int)
parser.add_argument("--num-updates", default=100, type=int)

parser.add_argument("--m0", default=-1.0, type=float)
parser.add_argument("--m1", default=1.0, type=float)
parser.add_argument("--s0", default=1.0, type=float)
parser.add_argument("--s1", default=1.0, type=float)
parser.add_argument("--p0", default=0.5, type=float)
parser.add_argument("--p1", default=0.5, type=float)


def train_update_loop(model_fn, n_train, n_update, n_test, num_updates, features, data_fn, seeds):
    seeds = np.arange(seeds)
    initial_fprs = {num_features: [] for num_features in features}
    updated_fprs = {num_features: [] for num_features in features}

    for seed in seeds:
        for num_features in features:
            np.random.seed(seed)

            x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                           num_features=num_features)

            model = model_fn(num_features=num_features)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            initial_tnr, initial_fpr, initial_fnr, initial_tpr = eval_model(y_test, y_pred)

            new_model, _ = update_model_feedback(model, x_update, y_update, None, None, num_updates)

            y_pred = new_model.predict(x_test)
            updated_tnr, updated_fpr, updated_fnr, updated_tpr = eval_model(y_test, y_pred)

            initial_fprs[num_features].append(initial_fpr)
            updated_fprs[num_features].append(updated_fpr)

    return initial_fprs, updated_fprs


def results_to_dataframe(initial_fprs, updated_fprs, features):
    data = {"type": [], "fpr": [], "num_features": []}

    for num_features in features:
        data["type"] += ["initial"] * len(initial_fprs[num_features])
        data["num_features"] += [num_features] * len(initial_fprs[num_features])
        data["fpr"] += initial_fprs[num_features]

        data["type"] += ["updated"] * len(updated_fprs[num_features])
        data["num_features"] += [num_features] * len(updated_fprs[num_features])
        data["fpr"] += updated_fprs[num_features]

    return pd.DataFrame(data)


def plot(data, plot_path):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sns.lineplot(x="num_features", y="fpr", hue="type", data=data, ax=ax)
    ax.set_xlabel("Num Features", size=20)
    ax.set_ylabel("FPR", size=20)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    fig.show()

    fig.savefig("{}.{}".format(plot_path, "png"), bbox_inches='tight', dpi=600)
    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    results_dir = os.environ.get("NUM_FEATURES_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    data_fn = get_data_fn(args)
    model_fn = get_model_fn(args.model)

    features = np.arange(args.features) + 2
    initial_fprs, updated_fprs = train_update_loop(model_fn, args.n_train, args.n_update, args.n_test,
                                                   args.num_updates, features, data_fn, args.seeds)

    data = results_to_dataframe(initial_fprs, updated_fprs, features)

    plot_name = "fpr_lineplot_num_features_{}".format(args.data_type)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    plot(data, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)




if __name__ == "__main__":
    main(parser.parse_args())