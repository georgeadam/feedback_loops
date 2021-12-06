import json
import os
import pickle

from typing import Dict

CHECKPOINT_FILE = "checkpoint.pt"
MODEL_FILE = "model.pt"
WEIGHT_FILE = "weights.pt"
RATE_FILE = "results.csv"
MASKS_FILE = "masks.pkl"
PREDICTION_FILE = "predictions.csv"

def create_file_path(file_path: str):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            print(OSError)


def save_json(results: Dict, file_path: str):
    create_file_path(file_path)

    with open(file_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


def save_pickle(results: Dict, file_path: str):
    create_file_path(file_path)

    with open(file_path, "wb") as fp:
        pickle.dump(results, fp)