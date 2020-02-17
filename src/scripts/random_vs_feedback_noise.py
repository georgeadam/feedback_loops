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
from src.utils.update import update_model_feedback, update_model_noise
from src.utils.rand import set_seed
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp


from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="gaussian", choices=["gaussian", "sklearn", "mimic"], type=str)
parser.add_argument("--seeds", default=10, type=int)
parser.add_argument("--model", default="lr", type=str)

parser.add_argument("--n-train", default=10000, type=float)
parser.add_argument("--n-update", default=10000, type=float)
parser.add_argument("--n-test", default=50000, type=float)
parser.add_argument("--num-features", default=2, type=int)
parser.add_argument("--num-updates", default=100, type=int)

parser.add_argument("--m0", default=-1.0, type=float)
parser.add_argument("--m1", default=1.0, type=float)
parser.add_argument("--s0", default=1.0, type=float)
parser.add_argument("--s1", default=1.0, type=float)
parser.add_argument("--p0", default=0.5, type=float)
parser.add_argument("--p1", default=0.5, type=float)


def train_update_loop(model_fn, n_train, n_update, n_test, num_updates, num_features, data_fn, seeds):
    seeds = np.arange(seeds)
    initial_fprs = []
    updated_fprs_noise = []
    updated_fprs_feedback = []

    for seed in seeds:
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)

        model = model_fn(num_features=num_features)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        initial_tnr, initial_fpr, initial_fnr, initial_tpr = eval_model(y_test, y_pred)

        new_model_noise = update_model_noise(model, x_update, y_update, num_updates, initial_fpr)
        new_model_feedback, _ = update_model_feedback(model, x_update, y_update, x_test, y_test, num_updates)

        y_pred_noise = new_model_noise.predict(x_test)
        updated_tnr_noise, updated_fpr_noise, updated_fnr_noise, updated_tpr_noise = eval_model(y_test, y_pred_noise)

        y_pred_feedback = new_model_feedback.predict(x_test)
        updated_tnr_feedback, updated_fpr_feedback, updated_fnr_feedback, updated_tpr_feedback = eval_model(y_test,
                                                                                                            y_pred_feedback)

        initial_fprs.append(initial_fpr)
        updated_fprs_noise.append(updated_fpr_noise)
        updated_fprs_feedback.append(updated_fpr_feedback)

    return initial_fprs, updated_fprs_noise, updated_fprs_feedback


def results_to_dataframe_boxplot(initial_fprs, updated_fprs_noise, updated_fprs_feedback):
    fprs = {"type": (["initial"] * len(initial_fprs)) + (["updated_noise"] * len(updated_fprs_noise)) + (
                ["updated_feedback"] * len(updated_fprs_feedback)),
                    "fpr": initial_fprs + updated_fprs_noise + updated_fprs_feedback}

    data = pd.DataFrame(fprs)

    return data


def results_to_dataframe_scatterplot(initial_fprs, updated_fprs_noise, updated_fprs_feedback):
    fprs = {"type": (["Noise"] * len(updated_fprs_noise)) + (["Feedback"] * len(updated_fprs_feedback)),
                        "initial_fpr": initial_fprs + initial_fprs,
                        "updated_fpr": updated_fprs_noise + updated_fprs_feedback}

    data = pd.DataFrame(fprs)

    return data


def boxplot(data, plot_path):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sns.boxplot(x="type", y="fpr", data=data, ax=ax)
    sns.swarmplot(x="type", y="fpr", data=data, color=".25", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("FPR", size=20)
    ax.set_title("Initial vs. Updated FPRs", size=26)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    fig.savefig("{}.{}".format(plot_path, "png"), bbox_inches='tight', dpi=600)
    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def scatterplot(data, plot_path):
    max_fpr = max(data["initial_fpr"].values.tolist() + data["updated_fpr"].values.tolist())
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    g = sns.scatterplot(x="initial_fpr", y="updated_fpr", hue="type", data=data, ax=ax, legend="full",
                        palette="bright")
    ax.set_xlabel("Initial FPR", size=30, labelpad=10.0)
    ax.set_ylabel("Updated FPR", size=30, labelpad=10.0)
    ax.set_xlim([0, max_fpr + 0.05])
    ax.set_ylim([0, max_fpr + 0.05])

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    legend = ax.legend(title="Update Type", title_fontsize=24, loc="lower right")
    legend.texts[0].set_size(20)
    legend.texts[0].set_visible(False)

    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    fig.savefig("{}.{}".format(plot_path, "png"), bbox_inches='tight', dpi=600)
    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    results_dir = os.environ.get("RANDOM_VS_FEEDBACK_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    data_fn = get_data_fn(args)
    model_fn = get_model_fn(args.model)

    initial_fprs, updated_fprs_noise, updated_fprs_feedback = train_update_loop(model_fn, args.n_train, args.n_update, args.n_test,
                                                   args.num_updates, args.num_features, data_fn, args.seeds)



    data_boxplot = results_to_dataframe_boxplot(initial_fprs, updated_fprs_noise, updated_fprs_feedback)

    plot_name = "fpr_boxplot_noise_vs_feedback_{}".format(args.data_type)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    boxplot(data_boxplot, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)

    data_scatterplot = results_to_dataframe_scatterplot(initial_fprs, updated_fprs_noise, updated_fprs_feedback)

    plot_name = "fpr_scatterplot_noise_vs_feedback_{}".format(args.data_type)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    scatterplot(data_scatterplot, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)


if __name__ == "__main__":
    main(parser.parse_args())