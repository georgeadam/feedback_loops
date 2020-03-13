import json


def load_json(file_path):
    with open(file_path, "r") as fp:
        res = json.load(fp)

    return res