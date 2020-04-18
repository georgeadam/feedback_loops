import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils.data import TEMPORAL_DATA_TYPES
from src.utils.update import map_update_type


def plot_rates_temporal(data, rate_types, update_types, title, plot_path):
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)

    if len(update_types) == 1:
        g = sns.lineplot(x="year", y="rate", hue="rate_type", data=data.loc[data["rate_type"].isin(rate_types)],
                         err_style="band", ax=ax ,ci="sd", palette="bright", marker="o")
    else:
        g = sns.lineplot(x="year", y="rate", hue="update_type", data=data.loc[data["rate_type"].isin(rate_types)],
                         err_style="band", ax=ax, style="rate_type" ,ci="sd", palette="bright", marker="o")

    ax.set_xlabel("Year", size=30, labelpad=10.0)
    ax.set_ylabel("Rate", size=30, labelpad=10.0)
    labels = []

    for i in range(len(g.lines)):
        label = g.lines[i].get_label()

        if len(update_types) == 1:
            if label in rate_types:
                labels.append(label.upper())
        else:
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
    if len(rate_types) == 1 and len(update_types) > 1:
        legend = ax.legend(title="Update Type", labels=labels, title_fontsize=30, borderaxespad=0.)
    elif len(rate_types) > 1 and len(update_types) == 1:
        legend = ax.legend(title="Rate Type", labels=labels, title_fontsize=30, borderaxespad=0.)
    else:
        legend = ax.legend(title="Rate Type", title_fontsize=30, borderaxespad=0.)

    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def plot_rates_static(data, rate_types, update_types, title, plot_path):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)

    if len(update_types) == 1:
        g = sns.lineplot(x="num_updates", y="rate", hue="rate_type", data=data.loc[data["rate_type"].isin(rate_types)],
                         err_style="band", ax=ax, ci="sd", palette="bright")
    else:
        g = sns.lineplot(x="num_updates", y="rate", hue="update_type", data=data.loc[data["rate_type"].isin(rate_types)],
                         err_style="band", ax=ax, ci="sd", style="rate_type", palette="bright")

    ax.set_xlabel("Num Updates", size=30, labelpad=10.0)
    ax.set_ylabel("Rate", size=30, labelpad=10.0)
    labels = []

    for i in range(len(g.lines)):
        label = g.lines[i].get_label()

        if len(update_types) == 1:
            if label in rate_types:
                labels.append(label.upper())
        else:
            if label in update_types:
                temp = map_update_type(label)
                temp = temp.replace("_", " ")
                labels.append(temp.upper())

    ax.set_xlim([0, np.max(data["num_updates"])])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    fig.suptitle(title)

    if len(rate_types) == 1 and len(update_types) > 1:
        legend = ax.legend(title="Update Type", labels=labels, title_fontsize=30, borderaxespad=0.)
    elif len(rate_types) > 1 and len(update_types) == 1:
        legend = ax.legend(title="Rate Type", labels=labels, title_fontsize=30, borderaxespad=0.)
    else:
        legend = ax.legend(title="Rate Type", title_fontsize=30, borderaxespad=0.)

    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def get_plot_fn(temporal):
    if temporal:
        return plot_rates_temporal
    else:
        return plot_rates_static