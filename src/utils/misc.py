from typing import Dict, List

def create_empty_rates() -> Dict[str, List]:
    return {"fpr": [], "tpr": [], "fnr": [], "tnr": [], "precision": [], "recall": [], "f1": [], "auc": [],
            "loss": [], "aupr": [], "fp_conf": [], "pos_conf": [], "fp_count": [], "total_samples": [],
            "fp_prop": [], "acc": []}


def capitalize(s: str) -> str:
    new_s = s[0].upper() + s[1:]

    return new_s