import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from typing import List

def plot_conditional_trust_static(data: pd.DataFrame, rate_types: List[str], model_fpr: float, title: str, plot_path: str):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)

    g1 = sns.lineplot(x="num_updates", y="rate", hue="clinician_fpr", data=data.loc[(data["rate_type"].isin(rate_types)) & (data["model_fpr"] == model_fpr)],
                     err_style="band", ax=ax, ci="sd", palette="bright", marker="o")

    ax.set_xlabel("Num Updates", size=30, labelpad=10.0)
    ax.set_ylabel(rate_types[0].upper(), size=30, labelpad=10.0)
    labels = []
    clinician_fprs = np.unique(data["clinician_fpr"])

    for i in range(len(g1.lines)):
        label = g1.lines[i].get_label()

        if label in clinician_fprs:
            if label == "0.0":
                labels.append("Blind Trust")
            else:
                labels.append("Conditional Trust")

    ax.set_xlim([0, np.max(data["num_updates"])])
    if rate_types[0] == "auc":
        ax.set_ylim([0.5, 1.0])
    else:
        ax.set_ylim([0.0, 1.0])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    ax.set_xlim([0, np.max(data["num_updates"])])
    fig.suptitle(title)

    legend = ax.legend(title="Trust Type", labels=labels, title_fontsize=30, borderaxespad=0.)
    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def plot_conditional_trust_temporal(data: pd.DataFrame, rate_types: List[str], model_fpr: float, title: str, plot_path: str):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)

    g1 = sns.lineplot(x="year", y="rate", hue="clinician_fpr", data=data.loc[(data["rate_type"].isin(rate_types)) & (data["model_fpr"] == model_fpr)],
                     err_style="band", ax=ax, ci="sd", palette="bright", marker="o")

    ax.set_xlabel("Year", size=30, labelpad=10.0)
    ax.set_ylabel(rate_types[0].upper(), size=30, labelpad=10.0)
    labels = []
    clinician_fprs = np.unique(data["clinician_fpr"])
    clinician_fprs = [str(clinician_fpr) for clinician_fpr in clinician_fprs]

    for i in range(len(g1.lines)):
        label = g1.lines[i].get_label()

        if label in clinician_fprs:
            if label == "0.0":
                labels.append("Blind Trust")
            else:
                labels.append("Conditional Trust")

    if rate_types[0] == "auc":
        ax.set_ylim([0.5, 1.0])
    else:
        ax.set_ylim([0.0, 1.0])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    ax.set_xticks(np.sort(data["year"].unique()))
    ax.set_xticklabels(np.sort(data["year"].unique()), rotation=90)

    fig.suptitle(title)

    legend = ax.legend(title="Trust Type", labels=labels, title_fontsize=30, borderaxespad=0.)
    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def plot_constant_trust_static(data: pd.DataFrame, rate_type: str, title: str, plot_path: str):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)

    g = sns.lineplot(x="num_updates", y="rate", hue="trust", data=data.loc[data["rate_type"] == rate_type],
                     err_style="band", ax=ax, ci="sd", palette="bright", marker="o")

    ax.set_xlabel("Num Updates", size=30, labelpad=10.0)
    ax.set_ylabel(rate_type.upper(), size=30, labelpad=10.0)
    labels = []
    trusts = np.unique(data["trust"])
    trusts = [str(trust) for trust in trusts]

    for i in range(len(g.lines)):
        label = g.lines[i].get_label()

        if label == "1.0":
            labels.append("Blind Trust")
        elif label in trusts:
            labels.append(label)

    ax.set_xlim([0, np.max(data["num_updates"])])
    if rate_type == "auc":
        ax.set_ylim([0.5, 1.0])
    else:
        ax.set_ylim([0.0, 1.0])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    fig.suptitle(title)

    legend = ax.legend(title="Trust", labels=labels, title_fontsize=30, borderaxespad=0.)
    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')


def plot_constant_trust_temporal(data: pd.DataFrame, rate_type: str, title: str, plot_path: str):
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111)

    g = sns.lineplot(x="year", y="rate", hue="trust", data=data.loc[data["rate_type"] == rate_type],
                     err_style="band", ax=ax, ci="sd", palette="bright", marker="o")

    ax.set_xlabel("Year", size=30, labelpad=10.0)
    ax.set_ylabel(rate_type.upper(), size=30, labelpad=10.0)
    labels = []
    trusts = np.unique(data["trust"])
    trusts = [str(trust) for trust in trusts]

    for i in range(len(g.lines)):
        label = g.lines[i].get_label()

        if label == "1.0":
            labels.append("Blind Trust")
        elif label in trusts:
            labels.append(label)

    if rate_type == "auc":
        ax.set_ylim([0.5, 1.0])
    else:
        ax.set_ylim([0.0, 1.0])

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)

    ax.set_xticks(np.sort(data["year"].unique()))
    ax.set_xticklabels(np.sort(data["year"].unique()), rotation=90)

    fig.suptitle(title)

    legend = ax.legend(title="Trust", labels=labels, title_fontsize=30, borderaxespad=0.)
    legend.texts[0].set_size(24)

    fig.savefig("{}.{}".format(plot_path, "pdf"), bbox_inches='tight')