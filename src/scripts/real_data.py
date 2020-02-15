import copy
import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from src.utils.data import generate_mimic_dataset, load_mimiciii_data
from src.utils.metrics import eval_model
from src.utils.model import get_model_fn
from src.utils.parse import percentage
from src.utils.update import update_model_feedback
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="mimic", choices=["mimic", "support2"], type=str)
parser.add_argument("--seeds", default=3, type=int)
parser.add_argument("--model", default="lr", type=str)

parser.add_argument("--n-train", default=0.4, type=percentage)
parser.add_argument("--n-update", default=0.4, type=percentage)
parser.add_argument("--n-test", default=0.2, type=percentage)
parser.add_argument("--num-updates", default=500, type=int)

parser.add_argument("--desired-fpr", default=0.1, type=float)
parser.add_argument("--rate-types", default=["fpr", "fnr"], nargs="+")


def train_update_loop(model_fn, n_train, n_update, n_test, names, num_updates, desired_fpr, data_fn, seeds):
    seeds = np.arange(seeds)

    rates = {name: [] for name in names}

    for seed in seeds:
        print(seed)
        np.random.seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test)

        model = model_fn()
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_train)

        threshold = find_threshold(y_train, y_prob, desired_fpr)

        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold
        initial_tnr, initial_fpr, initial_fnr, initial_tpr = eval_model(y_test, y_pred)

        new_model, temp_rates = update_model_feedback(model, x_update, y_update, x_test, y_test, num_updates,
                                                      intermediate=True, threshold=threshold)

        rates["fpr"].append([initial_fpr] + temp_rates["fpr"])
        rates["tpr"].append([initial_tpr] + temp_rates["tpr"])
        rates["fnr"].append([initial_fnr] + temp_rates["fnr"])
        rates["tnr"].append([initial_tnr] + temp_rates["tnr"])

    return rates


def gold_standard_loop(model_fn, n_train, n_update, n_test, names, desired_fpr, data_fn, seeds):
    seeds = np.arange(seeds)
    rates = {name: [] for name in names}

    for seed in seeds:
        np.random.seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test)

        model = model_fn()
        model.fit(np.concatenate((x_train, x_update)), np.concatenate((y_train, y_update)))
        y_prob = model.predict_proba(x_train)

        threshold = find_threshold(y_train, y_prob, desired_fpr)

        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold
        gold_standard_tnr, gold_standard_fpr, gold_standard_fnr, gold_standard_tpr = eval_model(y_test, y_pred)

        rates["fpr"].append(gold_standard_fpr)
        rates["tpr"].append(gold_standard_tpr)
        rates["fnr"].append(gold_standard_fnr)
        rates["tnr"].append(gold_standard_tnr)

    return rates


def find_threshold(y, y_prob, desired_fpr):
    thresholds = np.linspace(0.05, 0.95, 100)
    best_threshold = None
    best_fpr_diff = 0.0
    best_fpr = 0.0

    for threshold in thresholds:
        temp_pred = y_prob[:, 1] >= threshold
        temp_tnr, temp_fpr, temp_fnr, temp_tpr = eval_model(y, temp_pred)

        #         print("Threshold: {} | FPR: {}".format(threshold, temp_fpr))
        #         print("Best FPR diff: {} | (desired_fpr - temp_fpr) = {} | Current FPR: {}".format(best_fpr_diff, (desired_fpr - temp_fpr), temp_fpr))

        if temp_fpr < desired_fpr and best_threshold is None:
            best_threshold = threshold
            best_fpr_diff = desired_fpr - temp_fpr
            best_fpr = temp_fpr
        elif temp_fpr < desired_fpr and (desired_fpr - temp_fpr) < best_fpr_diff:
            best_threshold = threshold
            best_fpr_diff = desired_fpr - temp_fpr
            best_fpr = temp_fpr

    return best_threshold


def results_to_dataframe(rates):
    data = {"rate": [], "type": [], "num_updates": []}

    for name in rates.keys():
        for i in range(len(rates[name])):
            data["rate"] += rates[name][i]
            data["type"] += [name] * (len(rates[name][i]))
            data["num_updates"] +=  list(np.arange(len(rates[name][i])))

    return pd.DataFrame(data)


def plot_rates(data, rate_types, gs_lines, plot_path):
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)
    g = sns.lineplot(x="num_updates", y="rate", hue="type", data=data.loc[data["type"].isin(rate_types)],
                     err_style="band", ax=ax, ci="sd", palette="bright")

    ax.set_xlabel("Num Updates", size=30, labelpad=10.0)
    ax.set_ylabel("Rate", size=30, labelpad=10.0)

    for i in range(len(gs_lines)):
        ax.axhline(gs_lines[i], ls="--", color=g.lines[i].get_color())

    ax.set_xlim([0, 500])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    legend = ax.legend(title="Rate Type", title_fontsize=30, labels=list(map(lambda x: x.upper(), rate_types)),
                       loc="upper left")
    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "png"), bbox_inches='tight', dpi=600)
    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    results_dir = os.environ.get("REAL_DATA_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    if args.data_type == "mimic":
        data = load_mimiciii_data()
        data_fn = generate_mimic_dataset(data)

    n_train = int(len(data["y"]) * args.n_train)
    n_update = int(len(data["y"]) * args.n_update)
    n_test = int(len(data["y"]) * args.n_test)

    model_fn = get_model_fn(args.model)
    names = ["fpr", "tpr", "fnr", "tnr", "auc"]

    rates = train_update_loop(model_fn, n_train, n_update, n_test, names, args.num_updates,
                              args.desired_fpr, data_fn, args.seeds)
    gold_standard = gold_standard_loop(model_fn, n_train, n_update, n_test, names,
                                       args.desired_fpr, data_fn, args.seeds)

    data = results_to_dataframe(rates)



    plot_name = "{}_{}".format(args.data_type, args.rate_types)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    plot_rates(data, args.rate_types, [np.mean(gold_standard[key]) for key in args.rate_types], plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)



if __name__ == "__main__":
    main(parser.parse_args())