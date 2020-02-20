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
from src.utils.data import make_trend_gaussian_data
from src.utils.metrics import eval_model
from src.utils.update import update_model_feedback
from src.utils.save import create_file_path, save_json, CONFIG_FILE
from src.utils.time import get_timestamp

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--data-type", default="gaussian", type=str)

def train_update_loop():
    pass


def results_to_dataframe():
    pass


def plot(data, plot_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    fig.savefig("{}.{}".format(plot_path, "png"))
    fig.savefig("{}.{}".format(plot_path, "png"))


def main(args):
    load_dotenv(find_dotenv(), override=True)
    print(args)

    timestamp = get_timestamp()

    results_dir = os.environ.get("_RESULT_DIR")
    results_dir = os.path.join(ROOT_DIR, results_dir)

    results = None
    data = results_to_dataframe(results)
    config = args.__dict__



    plot_name = ""
    plot_file_name = "{}_{}".format(plot_name, timestamp)
    plot_path = os.path.join(results_dir, plot_file_name)

    create_file_path(plot_path)

    config_file_name = CONFIG_FILE.format(plot_name, timestamp)
    config_path = os.path.join(results_dir, config_file_name)

    save_json(config, config_path)



if __name__ == "__main__":
    main(parser.parse_args())