import copy
import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from src.utils.data import get_data_fn
from src.utils.metrics import eval_model
from src.utils.model import get_model_fn
from src.utils.parse import percentage
from src.utils.update import get_update_fn
from src.utils.rand import set_seed
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR
from sklearn.metrics import roc_auc_score, f1_score

parser = ArgumentParser()
parser.add_argument("--data-type", default="mimic", choices=["mimic", "support2", "gaussian"], type=str)
parser.add_argument("--seeds", default=3, type=int)
parser.add_argument("--model", default="random_forest", type=str)

parser.add_argument("--n-train", default=0.1, type=percentage)
parser.add_argument("--n-update", default=0.7, type=percentage)
parser.add_argument("--n-test", default=0.2, type=percentage)
parser.add_argument("--num-updates", default=500, type=int)
parser.add_argument("--num-features", default=20, type=int)

parser.add_argument("--desired-fpr", default=0.1, type=float)
parser.add_argument("--rate-types", default=["fpr", "fnr", "tpr", "tnr"], nargs="+")

parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--iterations", default=1000, type=int)
parser.add_argument("--importance", default=100000.0, type=float)

parser.add_argument("--update-type", default="feedback_full_fit", type=str)


def train_update_loop(model_fn, n_train, n_update, n_test, names, num_updates, num_features,
                      desired_fpr, data_fn, update_fn, seeds):
    seeds = np.arange(seeds)

    rates = {name: [] for name in names}

    initial_aucs = []
    updated_aucs = []

    initial_f1_scores = []
    updated_f1_scores = []

    for seed in seeds:
        print(seed)
        set_seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)

        model = model_fn(num_features=x_train.shape[1])
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_train)

        threshold = find_threshold(y_train, y_prob, desired_fpr)

        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold

        initial_tnr, initial_fpr, initial_fnr, initial_tpr = eval_model(y_test, y_pred)
        initial_auc = roc_auc_score(y_test, y_prob[:, 1])
        initial_f1_score = f1_score(y_test, y_pred)

        new_model, temp_rates = update_fn(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                          intermediate=True, threshold=threshold)

        y_prob = new_model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold
        updated_auc = roc_auc_score(y_test, y_prob[:, 1])
        updated_f1_score = f1_score(y_test, y_pred)

        rates["fpr"].append([initial_fpr] + temp_rates["fpr"])
        rates["tpr"].append([initial_tpr] + temp_rates["tpr"])
        rates["fnr"].append([initial_fnr] + temp_rates["fnr"])
        rates["tnr"].append([initial_tnr] + temp_rates["tnr"])
        initial_aucs.append(initial_auc)
        updated_aucs.append(updated_auc)

        initial_f1_scores.append(initial_f1_score)
        updated_f1_scores.append(updated_f1_score)

    return rates, initial_aucs, updated_aucs, initial_f1_scores, updated_f1_scores


def gold_standard_loop(model_fn, n_train, n_update, n_test, names, num_features, desired_fpr, data_fn, seeds):
    seeds = np.arange(seeds)
    rates = {name: [] for name in names}

    gold_standard_aucs = []
    gold_standard_f1_scores = []

    for seed in seeds:
        np.random.seed(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test,
                                                                       num_features=num_features)

        model = model_fn(num_features=x_train.shape[1])
        model.fit(np.concatenate((x_train, x_update)), np.concatenate((y_train, y_update)))
        y_prob = model.predict_proba(x_train)

        threshold = find_threshold(y_train, y_prob, desired_fpr)

        y_prob = model.predict_proba(x_test)
        y_pred = y_prob[:, 1] > threshold
        gold_standard_tnr, gold_standard_fpr, gold_standard_fnr, gold_standard_tpr = eval_model(y_test, y_pred)
        gold_standard_auc = roc_auc_score(y_test, y_prob[:, 1])
        gold_standard_f1_score = f1_score(y_test, y_pred)

        rates["fpr"].append(gold_standard_fpr)
        rates["tpr"].append(gold_standard_tpr)
        rates["fnr"].append(gold_standard_fnr)
        rates["tnr"].append(gold_standard_tnr)

        gold_standard_aucs.append(gold_standard_auc)
        gold_standard_f1_scores.append(gold_standard_f1_score)

    return rates, gold_standard_aucs, gold_standard_f1_scores


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


def plot_rates(data, rate_types, gs_lines, title, plot_path):
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

    fig.suptitle(title)

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

    results_dir = os.environ.get("DESIRED_FPR_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    data_fn = get_data_fn(args)
    model_fn = get_model_fn(args)
    update_fn = get_update_fn(args)

    names = ["fpr", "tpr", "fnr", "tnr", "auc"]

    rates, initial_aucs, updated_aucs, initial_f1_scores, updated_f1_scores = train_update_loop(model_fn, args.n_train,
                                                                                                args.n_update, args.n_test,
                                                                                                names, args.num_updates, args.num_features,
                                                                                                args.desired_fpr, data_fn, update_fn, args.seeds)
    gold_standard, gold_standard_aucs, gold_standard_f1_scores = gold_standard_loop(model_fn, args.n_train, args.n_update,
                                                                                    args.n_test, names, args.num_features,
                                                                                    args.desired_fpr, data_fn, args.seeds)

    data = results_to_dataframe(rates)

    plot_name = "{}_{}".format(args.data_type, args.rate_types)
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    plot_title = "Initial AUC: {} | Updated AUC: {} | Gold Standard AUC: {}".format(np.mean(initial_aucs),
                                                                                    np.mean(updated_aucs),
                                                                                    np.mean(gold_standard_aucs))

    create_file_path(plot_path)
    plot_rates(data, args.rate_types, [np.mean(gold_standard[key]) for key in args.rate_types], plot_title, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)

    print("Initial AUCS:")
    print(initial_aucs)
    print("Updated AUCS:")
    print(updated_aucs)

    print("Gold Standard AUCS:")
    print(gold_standard_aucs)

    print("Initial F1 Scores:")
    print(initial_f1_scores)
    print("Updated F1 Scores:")
    print(updated_f1_scores)

    print("Gold Standard F1 Scores:")
    print(gold_standard_f1_scores)



if __name__ == "__main__":
    main(parser.parse_args())