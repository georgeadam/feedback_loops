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
from src.utils.data import make_trend_gaussian_data, get_data_fn
from src.utils.metrics import eval_model, compute_all_rates
from src.utils.misc import create_empty_rates
from src.utils.model import get_model_fn
from src.utils.update import update_model_feedback, get_update_fn
from src.utils.rand import set_seed
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="moons", choices=["moons"], type=str)
parser.add_argument("--seeds", default=1, type=int)
parser.add_argument("--model", default="lr_pytorch", type=str)

parser.add_argument("--n-train", default=10000, type=int)
parser.add_argument("--n-update", default=10000, type=int)
parser.add_argument("--n-test", default=50000, type=int)
parser.add_argument("--num-features", default=2, type=int)
parser.add_argument("--num-updates", default=100, type=int)

parser.add_argument("--rate-types", default=["fpr"], nargs="+")

parser.add_argument("--lr", default=1.0, type=float)
parser.add_argument("--iterations", default=1000, type=int)
parser.add_argument("--importance", default=1.0, type=float)


parser.add_argument("--update-type", default="feedback", type=str)


def add_trend_to_data(x, y, num_updates, direction="pos"):
    batch_size = int(len(y) / float(num_updates))

    if num_updates > 1:
        percentages = np.linspace(0.0, 1.0, num_updates)
    else:
        percentages = [1.0]

    min_x1 = np.min(x[:, 0])
    min_x2 = np.min(x[:, 1])

    max_x1 = np.max(x[:, 0])
    max_x2 = np.max(x[:, 1])

    mean_x1 = np.mean(x[:, 0])
    mean_x2 = np.mean(x[:, 1])

    std_x1 = np.std(x[:, 0])
    std_x2 = np.std(x[:, 1])

    # for i in range(num_updates):

    for i in range(num_updates):
        if direction == "pos":
            idx = np.where(x[i * batch_size: (i + 1) * batch_size, 0] > (mean_x1 + std_x1))[0]

            idx = np.random.choice(idx, min(len(idx), int(batch_size * percentages[i])), replace=False)

            idx = idx + (i * batch_size)

            y[idx] = 0
        else:
            idx = np.where(x[i * batch_size: (i + 1) * batch_size, 0] < (mean_x1 - std_x1))[0]

            idx = np.random.choice(idx, min(len(idx), int(batch_size * percentages[i])), replace=False)

            idx = idx + (i * batch_size)

            y[idx] = 1

    return x, y

def train_update_loop(model_fn, n_train, n_update, n_test, num_updates, num_features, data_fn, update_fn, seeds, direction):
    seeds = np.arange(seeds)

    rates = {"updated_with_trend_on_shifted_data": create_empty_rates(), "updated_no_trend_on_shifted_data": create_empty_rates()}

    for seed in seeds:
        set_seed(seed)
        print(seed)

        x_train, y_train, x_update, y_update, x_test, y_test = data_fn(n_train, n_update, n_test, noise=0.3)
        x_update_no_trend, y_update_no_trend = x_update, y_update
        x_update_trend, y_update_trend = copy.deepcopy(x_update), copy.deepcopy(y_update)

        x_update_trend, y_update_trend = add_trend_to_data(x_update_trend, y_update_trend, num_updates, direction)
        x_test_shifted, y_test_shifted = add_trend_to_data(x_test, y_test, 1, direction)

        model = model_fn(num_features=num_features)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test_shifted)
        y_prob = model.predict_proba(x_test_shifted)
        initial_rates = compute_all_rates(y_test_shifted, y_pred, y_prob)

        new_model, rates_updated_no_trend_evaluated_trend = update_fn(model, x_train, y_train, x_update_no_trend,
                                                                      y_update_no_trend, x_test_shifted, y_test_shifted,
                                                                      num_updates, intermediate=True)

        new_model, rates_updated_trend_evaluated_trend = update_fn(model, x_train, y_train, x_update_trend,
                                                                   y_update_trend, x_test_shifted, y_test_shifted,
                                                                   num_updates, intermediate=True)

        for key in rates_updated_no_trend_evaluated_trend.keys():
            rates["updated_no_trend_on_shifted_data"][key].append([initial_rates[key]] +
                                                                       rates_updated_no_trend_evaluated_trend[key])
            rates["updated_with_trend_on_shifted_data"][key].append([initial_rates[key]] +
                                                                         rates_updated_trend_evaluated_trend[key])

    return rates


def results_to_dataframe(results, offset):
    data = {"display_type": [], "rate": [], "num_updates": [], "type": []}

    for stage in results.keys():
        for key in results[stage].keys():
            for i in range(len(results[stage][key])):
                if stage == "updated_no_trend_on_shifted_data":
                    data["display_type"] += ["Updated Model\nwith $P_{\mathrm{update}}(x)$ \nTested on " + "$Q^{{ {} }}".format(
                        offset) + "_{\mathrm{test}}(x)$" + " - {}".format(key)] * len(results[stage][key][i])
                elif stage == "updated_with_trend_on_shifted_data":
                    data["display_type"] += ["Updated Model\nwith" + "$Q^{{ {} }}".format(
                        offset) + "_{\mathrm{update}}(x)$ \nTested on " + "$Q^{{ {} }}".format(
                        offset) + "_{\mathrm{test}}(x)$" + " - {}".format(key)] * len(results[stage][key][i])

                data["type"] += [key] * len(results[stage][key][i])
                data["rate"] += results[stage][key][i]
                data["num_updates"] += np.arange(len(results[stage][key][i])).tolist()

    return pd.DataFrame(data)



def plot(data, rate_types, num_updates, plot_path):
    fig = plt.figure(figsize=(13, 9))

    ax = fig.add_subplot(111)

    g = sns.lineplot(x="num_updates", y="rate", hue="display_type", legend="full", data=data.loc[data["type"].isin(rate_types)],
                     ax=ax, palette="bright")

    ax.set_ylabel("Rate", fontsize=30, labelpad=10.0)
    ax.set_xlabel("Num Updates", fontsize=30, labelpad=10.0)

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    ax.set_xlim([0, num_updates])

    legend = ax.legend(title="Update Type", title_fontsize=30, loc="upper right",
                       bbox_to_anchor=(1.4, 1), borderaxespad=0., labelspacing=2.0)
    legend.texts[0].set_size(20)

    fig.show()
    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_extra_artists=(legend,), bbox_inches='tight')


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    config = args.__dict__

    timestamp = get_timestamp()

    results_dir = os.environ.get("CHANGING_P_Y_GIVEN_X_RESULTS_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    model_fn = get_model_fn(args)
    data_fn = get_data_fn(args)
    update_fn = get_update_fn(args)

    results_positive = train_update_loop(model_fn, args.n_train, args.n_update, args.n_test, args.num_updates,
                                         args.num_features, data_fn, update_fn, args.seeds, direction="pos")

    data_positive = results_to_dataframe(results_positive, "pos")

    results_negative = train_update_loop(model_fn, args.n_train, args.n_update, args.n_test, args.num_updates,
                                         args.num_features, data_fn, update_fn, args.seeds, direction="neg")

    data_negative = results_to_dataframe(results_negative, "neg")

    labels = ["Updated Model\nwith $P_{\mathrm{update}}(x)$ \nTested on $Q^{\mathrm{pos}}_{\mathrm{test}}(x)$"] + [
        "Updated Model\nwith $Q^{\mathrm{pos}}_{\mathrm{update}}(x)$ \nTested on $Q^{\mathrm{pos}}_{\mathrm{test}}(x)$"] + [
                 "Updated Model\nwith $P_{\mathrm{update}}(x)$ \nTested on $Q^{\mathrm{neg}}_{\mathrm{test}}(x)$"] + [
                 "Updated Model\nwith $Q^{\mathrm{neg}}_{\mathrm{update}}(x)$ \nTested on $Q^{\mathrm{neg}}_{\mathrm{test}}(x)$"]

    data = pd.concat([data_positive, data_negative])

    plot_name = "changed"
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)
    plot(data, args.rate_types, args.num_updates, plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)
    save_json(config, config_path)


if __name__ == "__main__":
    main(parser.parse_args())