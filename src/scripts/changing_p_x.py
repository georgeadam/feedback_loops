import copy
import importlib
import numpy as np
import pandas as pd
import os
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

import importlib

from src.models.sklearn import lr, linear_svm
from src.utils.data import make_trend_gaussian_data
from src.utils.metrics import eval_model
from src.utils.model import get_model_fn
from src.utils.update import update_model_feedback, update_model_feedback_linear_trend
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="gaussian", choices=["gaussian"], type=str)
parser.add_argument("--seeds", default=100, type=int)
parser.add_argument("--model", default="lr", type=str)

parser.add_argument("--n-train", default=10000, type=int)
parser.add_argument("--n-update", default=10000, type=int)
parser.add_argument("--n-test", default=50000, type=int)
parser.add_argument("--num-features", default=1, type=int)
parser.add_argument("--num-updates", default=100, type=int)

parser.add_argument("--m0", default=-1.0, type=float)
parser.add_argument("--m1", default=1.0, type=float)
parser.add_argument("--s0", default=1.0, type=float)
parser.add_argument("--s1", default=1.0, type=float)

parser.add_argument("--range-min", default=-5, type=float)
parser.add_argument("--range-max", default=5, type=float)

parser.add_argument("--offset", default=3.0, type=float)


def train_update_loop(model_fn, n_train, n_update, n_test, num_updates, m0, m1, s0, s1, num_features, uniform_range,
                      offset, seeds):
    seeds = np.arange(seeds)

    results = {"updated_with_trend_on_shifted_data_fprs": [], "updated_no_trend_on_shifted_data_fprs": []}

    for seed in seeds:
        np.random.seed(seed)
        print(seed)

        x_train, y_train = make_trend_gaussian_data(m0, m1, s0, s1, n_train, num_features, noise=0.0,
                                                    uniform_range=uniform_range)
        x_update, y_update = make_trend_gaussian_data(m0, m1, s0, s1, n_update, num_features, noise=0.0,
                                                      uniform_range=uniform_range)
        x_test_orig, y_test_orig = make_trend_gaussian_data(m0, m1, s0, s1, n_test, num_features, noise=0.0,
                                                            uniform_range=uniform_range)
        x_test_shifted, y_test_shifted = make_trend_gaussian_data(m0, m1, s0, s1, n_test, num_features, noise=0.0,
                                                                  uniform_range=[uniform_range[0] + offset,
                                                                                 uniform_range[1]])

        model = model_fn(num_features=num_features)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test_orig)

        y_pred = model.predict(x_test_shifted)
        initial_shifted_tnr, initial_shifted_fpr, initial_shifted_fnr, initial_shifted_tpr = eval_model(y_test_shifted,
                                                                                                        y_pred)

        new_model, rates_updated_no_trend_evaluated_trend = update_model_feedback(model, x_update, y_update,
                                                                                  x_test_shifted, y_test_shifted,
                                                                                  num_updates, intermediate=True)

        new_model, rates_update_trend_evaluated_trend = update_model_feedback_linear_trend(model, x_update, y_update,
                                                                                           x_test_shifted,
                                                                                           y_test_shifted, num_updates,
                                                                                           offset, intermediate=True)

        results["updated_no_trend_on_shifted_data_fprs"].append(
            [initial_shifted_fpr] + rates_updated_no_trend_evaluated_trend["fpr"])
        results["updated_with_trend_on_shifted_data_fprs"].append(
            [initial_shifted_fpr] + rates_update_trend_evaluated_trend["fpr"])

    return results


def results_to_dataframe(results, offset):
    data = {"type": [], "fpr": [], "num_updates": []}

    for t in results.keys():
        for i in range(len(results[t])):
            if t == "updated_no_trend_on_shifted_data_fprs":
                data["type"] += ["Updated Model\nwith $P_{\mathrm{update}}(x)$ \nTested on " + "$Q^{{ {} }}".format(
                    offset) + "_{\mathrm{test}}(x)$"] * len(results[t][i])
            elif t == "updated_with_trend_on_shifted_data_fprs":
                data["type"] += ["Updated Model\nwith" + "$Q^{{ {} }}".format(
                    offset) + "_{\mathrm{update}}(x)$ \nTested on " + "$Q^{{ {} }}".format(
                    offset) + "_{\mathrm{test}}(x)$"] * len(results[t][i])
            data["fpr"] += results[t][i]
            data["num_updates"] += np.arange(len(results[t][i])).tolist()

    return pd.DataFrame(data)


def plot(data, labels, num_updates, plot_path):
    fig = plt.figure(figsize=(13, 9))

    ax = fig.add_subplot(111)

    g = sns.lineplot(x="num_updates", y="fpr", hue="type", legend="full", data=data, ax=ax, palette="bright")

    ax.set_ylabel("FPR", fontsize=30, labelpad=10.0)
    ax.set_xlabel("Num Updates", fontsize=30, labelpad=10.0)

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    ax.set_xlim([0, num_updates])

    legend = ax.legend(title="Update Type", title_fontsize=30, loc="upper right", labels=labels,
                       bbox_to_anchor=(1.4, 1), borderaxespad=0., labelspacing=2.0)
    legend.texts[0].set_size(20)

    fig.show()
    fig.savefig("{}.{}".format(plot_path, "png"), bbox_extra_artists=(legend,), bbox_inches='tight', dpi=600)
    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_extra_artists=(legend,), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    results_dir = os.environ.get("CHANGING_P_X_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    model_fn = get_model_fn(args.model)

    uniform_range = [args.range_min, args.range_max]
    offset = args.offset
    results_positive = train_update_loop(model_fn, args.n_train, args.n_update, args.n_test, args.num_updates,
                                                            args.m0, args.m1, args.s0, args.s1, args.num_features,
                                                            uniform_range, offset, args.seeds)

    data_positive = results_to_dataframe(results_positive, "pos")

    offset = - args.offset
    results_negative = train_update_loop(model_fn, args.n_train, args.n_update, args.n_test, args.num_updates,
                                         args.m0, args.m1, args.s0, args.s1, args.num_features,
                                         uniform_range, offset, args.seeds)
    data_negative = results_to_dataframe(results_negative, "neg")

    labels = ["Updated Model\nwith $P_{\mathrm{update}}(x)$ \nTested on $Q^{\mathrm{pos}}_{\mathrm{test}}(x)$"] + [
        "Updated Model\nwith $Q^{\mathrm{pos}}_{\mathrm{update}}(x)$ \nTested on $Q^{\mathrm{pos}}_{\mathrm{test}}(x)$"] + [
                 "Updated Model\nwith $P_{\mathrm{update}}(x)$ \nTested on $Q^{\mathrm{neg}}_{\mathrm{test}}(x)$"] + [
                 "Updated Model\nwith $Q^{\mathrm{neg}}_{\mathrm{update}}(x)$ \nTested on $Q^{\mathrm{neg}}_{\mathrm{test}}(x)$"]

    data = pd.concat([data_positive, data_negative])

    plot_name = "linear_trend"
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    plot(data, labels, args.num_updates, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)


if __name__ == "__main__":
    main(parser.parse_args())