from src.utils.save import CONFIG_FILE, STATS_FILE



def create_empty_rates():
    return {"fpr": [], "tpr": [], "fnr": [], "tnr": [], "precision": [], "recall": [], "f1": [], "auc": [],
            "loss": [], "aupr": [], "fp_conf": [], "pos_conf": []}


def capitalize(s):
    new_s = s[0].upper() + s[1:]

    return new_s


def create_config_file_name(file_name, plot_name, timestamp):
    if file_name == "timestamp":
        config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    else:
        config_file_name = CONFIG_FILE.format(plot_name, "")

    return config_file_name


def create_plot_file_name(file_name, plot_name, timestamp):
    if file_name == "timestamp":
        plot_file_name = "{}_{}".format(plot_name, timestamp)
    else:
        plot_file_name = "{}_{}".format(plot_name, "")

    return plot_file_name


def create_stats_file_name(file_name, plot_name, timestamp):
    if file_name == "timestamp":
        stats_file_name = STATS_FILE.format(plot_name,  timestamp)
    else:
        stats_file_name = STATS_FILE.format(plot_name, "")

    return stats_file_name