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
from src.utils.update import get_update_fn
from src.utils.save import create_file_path, save_json, CONFIG_FILE, STATS_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="mimic", choices=["mimic", "support2", "gaussian"], type=str)
parser.add_argument("--seeds", default=1, type=int)
parser.add_argument("--model", default="nn", type=str)

parser.add_argument("--n-train", default=0.1, type=percentage)
parser.add_argument("--n-update", default=0.7, type=percentage)
parser.add_argument("--n-test", default=0.2, type=percentage)
parser.add_argument("--num-updates", default=500, type=int)
parser.add_argument("--num-features", default=20, type=int)

parser.add_argument("--initial-desired-rate", default="fpr", type=str)
parser.add_argument("--initial-desired-value", default=0.1, type=float)

parser.add_argument("--dynamic-desired-rate", default=None, type=str)

parser.add_argument("--rate-types", default=["auc", "fpr", "fnr", "tnr"], nargs="+")

parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--iterations", default=2500, type=int)
parser.add_argument("--importance", default=100000.0, type=float)

parser.add_argument("--hidden-layers", default=0, type=int)
parser.add_argument("--activation", default="Tanh", type=str)

parser.add_argument("--bad-model", default=False, type=str2bool)
parser.add_argument("--update-type", default="feedback_online_single_batch", type=str)


def results_to_dataframe(rates):
    data = {"rate": [], "type": [], "num_updates": []}

    for name in rates.keys():
        for i in range(len(rates[name])):
            data["rate"] += rates[name][i]
            data["type"] += [name] * (len(rates[name][i]))
            data["num_updates"] +=  list(np.arange(len(rates[name][i])))

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


def plot_rates(data, rate_types, gs_lines, title, plot_path):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)
    g = sns.lineplot(x="num_updates", y="rate", hue="type", data=data.loc[data["type"].isin(rate_types)],
                     err_style="band", ax=ax, ci="sd", palette="bright")

    ax.set_xlabel("Num Updates", size=30, labelpad=10.0)
    ax.set_ylabel("Rate", size=30, labelpad=10.0)
    labels = []

    for i in range(len(g.lines)):
        label = g.lines[i].get_label()

        if label in gs_lines.keys():
            labels.append(label.upper())
            ax.axhline(gs_lines[label], ls="--", color=g.lines[i].get_color())

    ax.set_xlim([0, 500])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    fig.suptitle(title)

    # legend = ax.legend(title="Rate Type", labels=labels, title_fontsize=30,
    #                    loc="upper right", bbox_to_anchor=(1.25, 1), borderaxespad=0.)
    legend = ax.legend(title="Rate Type", labels=labels, title_fontsize=30,
                       loc="upper right")
    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    results_dir = os.environ.get("DESIRED_FPR_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    data_fn = get_data_fn(args)
    model_fn = get_model_fn(args)
    update_fn = get_update_fn(args)

    rates, stats = train_update_loop(model_fn, args.n_train, args.n_update,
                                     args.n_test, args.num_updates, args.num_features,
                                     args.initial_desired_rate, args.initial_desired_value, args.dynamic_desired_rate,
                                     data_fn, update_fn, args.bad_model,
                                     args.seeds)
    gold_standard = gold_standard_loop(model_fn, args.n_train, args.n_update,
                                       args.n_test, args.num_features,
                                       args.initial_desired_rate, args.initial_desired_value, data_fn, args.seeds)

    data = results_to_dataframe(rates)
    stats["gold_standard"] = gold_standard
    stats = summarize_stats(stats)

    plot_name = "{}_{}".format(args.data_type, args.rate_types)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    plot_title = ""

    create_file_path(plot_path)
    plot_rates(data, args.rate_types, {key: np.mean(gold_standard[key]) for key in args.rate_types}, plot_title, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)

    stats_file_name = STATS_FILE.format(plot_name,  timestamp)
    stats_path = os.path.join(results_dir, stats_file_name)
    save_json(stats, stats_path)


if __name__ == "__main__":
    main(parser.parse_args())