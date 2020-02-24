import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from src.utils.data import get_data_fn
from src.utils.misc import create_empty_rates
from src.utils.metrics import eval_model, compute_all_rates
from src.utils.model import get_model_fn
from src.utils.parse import percentage, str2bool
from src.utils.update import get_update_fn, find_threshold
from src.utils.rand import set_seed
from src.utils.save import create_file_path, save_json, CONFIG_FILE, STATS_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

parser = ArgumentParser()
parser.add_argument("--data-type", default="mimic", choices=["mimic", "support2", "gaussian"], type=str)
parser.add_argument("--seeds", default=10, type=int)
parser.add_argument("--model", default="nn", type=str)

parser.add_argument("--n-train", default=0.1, type=percentage)
parser.add_argument("--n-update", default=0.7, type=percentage)
parser.add_argument("--n-test", default=0.2, type=percentage)
parser.add_argument("--num-updates", default=500, type=int)
parser.add_argument("--num-features", default=20, type=int)

parser.add_argument("--initial-desired-rate", default="fpr", type=str)
parser.add_argument("--initial-desired-value", default=0.1, type=float)

parser.add_argument("--dynamic-desired-rate", default="fnr", type=str)

parser.add_argument("--rate-types", default=["precision", "recall", "fpr", "fnr"], nargs="+")

parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--iterations", default=1000, type=int)
parser.add_argument("--importance", default=100000.0, type=float)

parser.add_argument("--bad-model", default=False, type=str2bool)
parser.add_argument("--update-type", default="no_feedback", type=str)


def train_update_loop(model_fn, n_train, n_update, n_test, num_updates, num_features,
                      initial_desired_rate, initial_desired_value, dynamic_desired_rate, data_fn, update_fn, bad_model,
                      seeds):
    seeds = np.arange(seeds)

    rates = create_empty_rates()

    stats = {"updated": {key: [] for key in rates.keys()},
             "initial": {key: [] for key in rates.keys()}}

    for seed in seeds:
        print(seed)
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)

        model = model_fn(num_features=x_train.shape[1])

        if not bad_model:
            model.fit(x_train, y_train)

        y_prob = model.predict_proba(x_train)
        threshold = find_threshold(y_train, y_prob, initial_desired_rate, initial_desired_value)

        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold

        initial_rates = compute_all_rates(y_test, y_pred, y_prob)
        dynamic_desired_value = get_dyanmic_desired_value(dynamic_desired_rate, initial_rates)

        new_model, updated_rates = update_fn(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                          intermediate=True, threshold=threshold,
                                          dynamic_desired_rate=dynamic_desired_rate,
                                          dynamic_desired_value=dynamic_desired_value)

        for key in rates.keys():
            rates[key].append([initial_rates[key]] + updated_rates[key])
            stats["initial"][key].append(initial_rates[key])
            stats["updated"][key].append(updated_rates[key][-1])

    return rates, stats


def gold_standard_loop(model_fn, n_train, n_update, n_test, num_features, desired_rate, desired_value, data_fn, seeds):
    seeds = np.arange(seeds)
    rates = create_empty_rates()

    for seed in seeds:
        np.random.seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)

        model = model_fn(num_features=x_train.shape[1])
        model.fit(np.concatenate((x_train, x_update)), np.concatenate((y_train, y_update)))
        y_prob = model.predict_proba(np.concatenate((x_train, x_update)))

        threshold = find_threshold(np.concatenate((y_train, y_update)), y_prob, desired_rate, desired_value)

        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold
        gold_standard_rates = compute_all_rates(y_test, y_pred, y_prob)

        for key in rates.keys():
            rates[key].append(gold_standard_rates[key])

    return rates


def get_dyanmic_desired_value(desired_dynamic_rate, rates):
    if desired_dynamic_rate is not None:
        return rates[desired_dynamic_rate]

    return None


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
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)
    g = sns.lineplot(x="num_updates", y="rate", hue="type", data=data.loc[data["type"].isin(rate_types)],
                     err_style="band", ax=ax, ci="sd", palette="bright")

    ax.set_xlabel("Num Updates", size=30, labelpad=10.0)
    ax.set_ylabel("Rate", size=30, labelpad=10.0)

    for i in range(len(g.lines)):
        label = g.lines[i].get_label()

        if label in gs_lines.keys():
            ax.axhline(gs_lines[label], ls="--", color=g.lines[i].get_color())

    ax.set_xlim([0, 500])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    fig.suptitle(title)

    legend = ax.legend(title="Rate Type", title_fontsize=30,
                       loc="upper left")
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