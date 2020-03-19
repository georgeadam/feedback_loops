import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV

sns.set_style("white")
import matplotlib.pyplot as plt

from src.scripts.helpers.generic import get_dyanmic_desired_value
from src.models.sklearn import evaluate
from src.utils.data import get_data_fn
from src.utils.misc import capitalize
from src.utils.model import get_model_fn
from src.utils.parse import percentage, str2bool
from src.utils.update import get_update_fn, map_update_type
from src.utils.save import create_file_path, save_json, CONFIG_FILE, STATS_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="mimic_iv", choices=["mimic_iii", "mimic_iv", "support2", "gaussian"], type=str)
parser.add_argument("--seeds", default=1, type=int)
parser.add_argument("--model", default="xgboost", type=str)
parser.add_argument("--warm-start", default=False, type=str2bool)

parser.add_argument("--num-features", default=20, type=int)
parser.add_argument("--train-year-limit", default=1999, type=int)
parser.add_argument("--update-year-limit", default=2019, type=int)
parser.add_argument("--next-year", default=False, type=str2bool)
parser.add_argument("--sorted", default=False, type=str2bool)

parser.add_argument("--initial-desired-rate", default="fpr", type=str)
parser.add_argument("--initial-desired-value", default=0.1, type=float)

parser.add_argument("--dynamic-desired-rate", default=None, type=str)

parser.add_argument("--rate-types", default=["auc"], nargs="+")
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
parser.add_argument("--update-types", default=["feedback_full_fit",
                                               "no_feedback_full_fit",
                                               "feedback_full_fit_confidence",
                                               "no_feedback_full_fit_confidence",
                                               "evaluate"], type=str)


parser.add_argument("--save-dir", default="figures/paper/figure_1", type=str)


from src.utils.metrics import compute_all_rates
from src.utils.misc import create_empty_rates
from src.utils.rand import set_seed
from src.utils.update import find_threshold


def train_update_loop(model_fn, train_year_limit, update_year_limit, initial_desired_rate, initial_desired_value,
                      dynamic_desired_rate, data_fn, update_fn, bad_model, next_year, seeds):
    seeds = np.arange(seeds)

    rates = create_empty_rates()

    stats = {"updated": {key: [] for key in rates.keys()},
             "initial": {key: [] for key in rates.keys()}}

    for seed in seeds:
        print(seed)
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(0.3, 0.4, 0.3, num_features=0)
        x = np.concatenate([x_train, x_update, x_test], axis=0)
        y = np.concatenate([y_train, y_update, y_test])

        model = model_fn(num_features=x.shape[1] - 1)
        train_idx = x[:, 0] <= train_year_limit
        x_train, y_train = x[train_idx], y[train_idx]
        x_rest, y_rest = x[~train_idx], y[~train_idx]

        if next_year:
            eval_idx = x[:, 0] == train_year_limit + 1
        else:
            eval_idx = x[:, 0] == update_year_limit

        x_eval, y_eval = x[eval_idx],  y[eval_idx]

        if not bad_model:
            model.fit(x_train[:, 1:], y_train)
            loss = model.evaluate(x_eval[:, 1:], y_eval)

        y_prob = model.predict_proba(x_train[:, 1:])

        if initial_desired_rate is not None:
            threshold = find_threshold(y_train, y_prob, initial_desired_rate, initial_desired_value)
        else:
            threshold = 0.5

        y_prob = model.predict_proba(x_eval[:, 1:])
        y_pred = y_prob[:, 1] > threshold

        initial_rates = compute_all_rates(y_eval, y_pred, y_prob)
        initial_rates["loss"] = loss

        y_prob = model.predict_proba(x_train[:, 1:])
        y_pred = y_prob[:, 1] > threshold
        temp_train_rates = compute_all_rates(y_train, y_pred, y_prob)
        dynamic_desired_value = get_dyanmic_desired_value(dynamic_desired_rate, temp_train_rates)

        years = x_rest[:, 0]
        x_train = np.delete(x_train, 0, 1)
        x_rest = np.delete(x_rest, 0, 1)

        new_model, updated_rates = update_fn(model, x_train, y_train, x_rest, y_rest, years, train_year_limit, update_year_limit,
                                             next_year=next_year,
                                             intermediate=True, threshold=threshold,
                                             dynamic_desired_rate=dynamic_desired_rate,
                                             dynamic_desired_value=dynamic_desired_value)

        for key in rates.keys():
            rates[key].append([initial_rates[key]] + updated_rates[key])
            stats["initial"][key].append(initial_rates[key])
            stats["updated"][key].append(updated_rates[key][-1])

    return rates, stats


def results_to_dataframe(rates, train_year_limit, update_year_limit):
    data = {"rate": [], "rate_type": [], "year": [], "update_type": []}

    for update_type in rates.keys():
        for name in rates[update_type].keys():
            if name != "loss":
                for i in range(len(rates[update_type][name])):
                    data["rate"] += rates[update_type][name][i]
                    data["rate_type"] += [name] * (len(rates[update_type][name][i]))
                    data["year"] +=  list(np.arange(train_year_limit, update_year_limit))
                    data["update_type"] += [update_type] * (len(rates[update_type][name][i]))

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


def plot_rates(data, rate_types, update_types, title, plot_path):
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)
    g = sns.lineplot(x="year", y="rate", hue="update_type", data=data.loc[data["rate_type"].isin(rate_types)],
                     err_style="band", ax=ax, ci="sd", palette="bright", marker="o")

    ax.set_xlabel("Year", size=30, labelpad=10.0)
    ax.set_ylabel(rate_types[0].upper(), size=30, labelpad=10.0)
    labels = []

    for i in range(len(g.lines)):
        label = g.lines[i].get_label()

        if label in update_types:
            temp = map_update_type(label)
            temp = temp.replace("_", " ")
            labels.append(temp.upper())

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    ax.set_xticks(np.sort(data["year"].unique()))
    ax.set_xticklabels(np.sort(data["year"].unique()), rotation=90)

    fig.suptitle(title)

    # legend = ax.legend(title="Rate Type", labels=labels, title_fontsize=30,
    #                    loc="upper right", bbox_to_anchor=(1.30, 1), borderaxespad=0.)
    legend = ax.legend(title="Rate Type", labels=labels, title_fontsize=30,
                       loc="lower left", borderaxespad=0.)

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
        results_dir = os.environ.get("DESIRED_FPR_RESULTS_DIR")

    results_dir = os.path.join(ROOT_DIR, results_dir)

    data_fn = get_data_fn(args)
    model_fn = get_model_fn(args)

    rates = {}
    stats = {}

    for update_type in args.update_types:
        update_fn = get_update_fn(update_type, temporal=True)
        temp_rates, temp_stats = train_update_loop(model_fn, args.train_year_limit, args.update_year_limit,
                                         args.initial_desired_rate, args.initial_desired_value,
                                         args.dynamic_desired_rate,
                                         data_fn, update_fn, args.bad_model, args.next_year,
                                         args.seeds)
        rates[update_type] = temp_rates
        stats[update_type] = temp_stats

    data = results_to_dataframe(rates, args.train_year_limit, args.update_year_limit)
    for update_type in args.update_types:
        stats[update_type] = summarize_stats(stats[update_type])

    plot_name = "{}_{}".format(args.data_type, args.rate_types)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    plot_title = ""

    create_file_path(plot_path)
    plot_rates(data, args.rate_types, args.update_types, plot_title, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)

    stats_file_name = STATS_FILE.format(plot_name,  timestamp)
    stats_path = os.path.join(results_dir, stats_file_name)
    save_json(stats, stats_path)


if __name__ == "__main__":
    main(parser.parse_args())