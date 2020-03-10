import numpy as np
import os
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from src.utils.data import get_data_fn
from src.utils.load import load_json
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

import shutil

parser = ArgumentParser()
parser.add_argument("--data-type", default="mimic", choices=["mimic", "support2", "gaussian"], type=str)
parser.add_argument("--seeds", default=5, type=int)
parser.add_argument("--model", default="nn", type=str)

parser.add_argument("--n-train", default=0.1, type=percentage)
parser.add_argument("--n-update", default=0.7, type=percentage)
parser.add_argument("--n-test", default=0.2, type=percentage)
parser.add_argument("--num-updates", default=250, type=int)
parser.add_argument("--num-features", default=20, type=int)

parser.add_argument("--initial-desired-rate", default="fpr", type=str)
parser.add_argument("--initial-desired-value", default=0.1, type=float)

parser.add_argument("--dynamic-desired-rate", default=None, type=str)

parser.add_argument("--rate-types", default=["auc", "fpr", "fnr"], nargs="+")

# parser.add_argument("--lr", default=0.1, type=float)
# parser.add_argument("--iterations", default=1000, type=int)
parser.add_argument("--importance", default=100000.0, type=float)

parser.add_argument("--hidden-layers", default=0, type=int)
parser.add_argument("--activation", default="Tanh", type=str)

parser.add_argument("--bad-model", default=False, type=str2bool)
parser.add_argument("--update-type", default="feedback", type=str)


def compare_configs(config1, config2):
    for key in config1.keys():
        if key not in config2 or config1[key] != config2[key]:
            return False

    return True


def get_img(image_files, timestamp):
    for img_file in image_files:
        if img_file.startswith(timestamp):
            return img_file

    return None


def find_all_models_with_config(models_dir, config):
    img_files = [d for d in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, d)) and d.endswith(".pdf")]
    config_files = [d for d in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, d)) and d.endswith("config.json")]
    models = []

    for f in config_files:
        config_path = os.path.join(models_dir, f)

        temp_config = load_json(config_path)

        if compare_configs(config, temp_config):
            img = get_img(img_files, f.split("_config.json")[0])
            # print("Found model with specified config in config: {}".format(config_path))
            models.append(f)
            models.append(img)

    return models


def main(args):
    d = os.path.join(ROOT_DIR, "figures/desired_fpr")
    m_d = os.path.join(ROOT_DIR, "figures/temp")

    models = find_all_models_with_config(d, args.__dict__)
    print(models)

    for f in models:
        shutil.copyfile(os.path.join(d, f), os.path.join(m_d, f))


if __name__ == "__main__":
    main(parser.parse_args())