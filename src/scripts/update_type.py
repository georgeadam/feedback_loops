import numpy as np
import os
import pandas as pd
import seaborn as sns

from src.scripts.helpers.generic import train_update_loop, gold_standard_loop

sns.set()
import matplotlib.pyplot as plt

from src.utils.data import get_data_fn
from src.utils.model import get_model_fn
from src.utils.parse import percentage, str2bool
from src.utils.update import get_update_fn, map_update_type
from src.utils.save import create_file_path, save_json, CONFIG_FILE, STATS_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="mimic_iii", choices=["mimic_iii", "mimic_iv", "support2", "gaussian"], type=str)
parser.add_argument("--seeds", default=10, type=int)
parser.add_argument("--model", default="nn", type=str)

parser.add_argument("--n-train", default=0.1, type=percentage)
parser.add_argument("--n-update", default=0.7, type=percentage)
parser.add_argument("--n-test", default=0.2, type=percentage)
parser.add_argument("--num-updates", default=100, type=int)
parser.add_argument("--num-features", default=20, type=int)

parser.add_argument("--initial-desired-rate", default="fpr", type=str)
parser.add_argument("--initial-desired-value", default=0.1, type=float)

parser.add_argument("--dynamic-desired-rate", default=None, type=str)

parser.add_argument("--rate-types", default=["fpr"], nargs="+")
# parser.add_argument("--rate-types", default=["loss"], nargs="+")

parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--online-lr", default=0.01, type=float)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--reset-optim", default=False, type=str2bool)
parser.add_argument("--iterations", default=3000, type=int)
parser.add_argument("--importance", default=100000.0, type=float)
parser.add_argument("--tol", default=0.0001, type=float)

parser.add_argument("--hidden-layers", default=0, type=int)
parser.add_argument("--activation", default="Tanh", type=str)

parser.add_argument("--bad-model", default=False, type=str2bool)
parser.add_argument("--worst-case", default=False, type=str2bool)
parser.add_argument("--update-types", default=["feedback_online_single_batch",
                                               "feedback_online_all_update_data",
                                               "feedback_online_all_data",
                                               "feedback_full_fit",
                                               "no_feedback_full_fit"], type=str)

parser.add_argument("--save-dir", default="figures/paper/update_type", type=str)


def results_to_dataframe(rates):
    data = {"rate": [], "rate_type": [], "num_updates": [], "update_type": []}

    for update_type in rates.keys():
        for name in rates[update_type].keys():
            for i in range(len(rates[update_type][name])):
                data["rate"] += rates[update_type][name][i]
                data["rate_type"] += [name] * len(rates[update_type][name][i])
                data["num_updates"] +=  list(np.arange(len(rates[update_type][name][i])))
                data["update_type"] += [update_type] * len(rates[update_type][name][i])

    return pd.DataFrame(data)


def summarize_stats(stats):
    metrics = ["median", "mean", "std", "min", "max"]
    summary = {stage: {key: {} for key in stats[stage].keys()} for stage in stats.keys()}

    for stage in stats.keys():
        for key in stats[stage].keys():
            for metric in metrics:
                fn = getattr(np, metric)
                res = fn(stats[stage][key])

                summary[stage][key][metric] = res

    return summary


def plot_rates(data, rate_types, update_types, num_updates, title, plot_path):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    g = sns.lineplot(x="num_updates", y="rate", hue="update_type", data=data.loc[data["rate_type"].isin(rate_types)],
                     err_style="band", ax=ax, ci="sd", palette="bright")

    ax.set_xlabel("Num Updates", size=30, labelpad=10.0)
    ax.set_ylabel("FPR", size=30, labelpad=10.0)
    labels = []

    for i in range(len(g.lines)):
        label = g.lines[i].get_label()

        if label in update_types:
            temp = map_update_type(label)
            temp = temp.replace("_", " ")
            labels.append(temp.upper())

    ax.set_xlim([0, num_updates])
    ax.set_ylim([0, 1.0])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    fig.suptitle(title)

    # legend = ax.legend(title="Rate Type", labels=labels, title_fontsize=30,
    #                    loc="upper right", bbox_to_anchor=(1.3, 1), borderaxespad=0.)
    legend = ax.legend(title="Update Type", labels=labels, title_fontsize=30,
                       loc="upper left")
    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    if args.save_dir is not None:
        results_dir = args.save_dir
    else:
        results_dir = os.environ.get("UPDATE_TYPE_COMPARISON_RESULTS_DIR")

    results_dir = os.path.join(ROOT_DIR, results_dir)

    data_fn = get_data_fn(args)
    model_fn = get_model_fn(args)
    rates = {}

    for update_type in args.update_types:
        update_fn = get_update_fn(update_type)
        temp_rates, _ = train_update_loop(model_fn, args.n_train, args.n_update,
                                         args.n_test, args.num_updates, args.num_features,
                                         args.initial_desired_rate, args.initial_desired_value, args.dynamic_desired_rate,
                                         data_fn, update_fn, args.bad_model, args.worst_case,
                                         args.seeds)
        rates[update_type] = temp_rates

    data = results_to_dataframe(rates)

    plot_name = "{}_{}".format(args.data_type, args.rate_types)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    plot_title = ""

    create_file_path(plot_path)
    plot_rates(data, args.rate_types, args.update_types, args.num_updates, plot_title, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)


if __name__ == "__main__":
    main(parser.parse_args())