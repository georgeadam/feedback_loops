import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_conditional_trust_static(data, rate_types, model_fpr, title, plot_path):
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


def plot_conditional_trust_temporal(data, rate_types, model_fpr, title, plot_path):
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


def plot_constant_trust_static(data, rate_type, title, plot_path):
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


def plot_constant_trust_temporal(data, rate_type, title, plot_path):
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


def get_plot_fn(temporal, trust_type):
    if temporal and trust_type == "conditional":
        return plot_conditional_trust_temporal
    elif temporal and trust_type == "constant":
        return plot_constant_trust_temporal
    elif not temporal and trust_type == "conditional":
        return plot_conditional_trust_static
    else:
        return plot_constant_trust_static