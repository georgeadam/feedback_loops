def create_empty_rates():
    return {"fpr": [], "tpr": [], "fnr": [], "tnr": [], "precision": [], "recall": [], "f1": [], "auc": [],
            "loss": []}


def capitalize(s):
    new_s = s[0].upper() + s[1:]

    return new_s