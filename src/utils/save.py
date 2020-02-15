import json
import os
import pickle

CHECKPOINT_FILE = "checkpoint.pt"
MODEL_FILE = "model.pt"
WEIGHT_FILE = "weights.pt"
CONFIG_FILE = "{}_{}_config.json"
STATS_FILE = "stats.json"
MASKS_FILE = "masks.pkl"

def create_file_path(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            print(OSError)


def save_json(results, file_path):
    create_file_path(file_path)

    with open(file_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


def save_pickle(results, file_path):
    create_file_path(file_path)

    with open(file_path, "wb") as fp:
        pickle.dump(results, fp)