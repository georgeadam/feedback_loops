from typing import Any
import numpy as np


def full_trust(*args: Any, **kwargs: Any) -> float:
    return 1.0


def conditional_trust(model_fpr: float=0.0, **kwargs: Any) -> float:
    return 1 - model_fpr


def constant_trust(clinician_trust: float=1.0, **kwargs: Any) -> float:
    return clinician_trust


def confidence_trust(model_prob: np.array, **kwargs: Any) -> float:
    return model_prob


def confidence_threshold_trust(model_prob: np.array, **kwargs: Any) -> float:
    drop_idx = model_prob < 0.9

    temp = np.ones(len(model_prob)).astype(float)
    temp[drop_idx] = 0

    return temp


def get_trust_fn(trust_fn_name):
    if trust_fn_name == "full_trust":
        return full_trust
    elif trust_fn_name == "conditional_trust":
        return conditional_trust
    elif trust_fn_name == "constant_trust":
        return constant_trust
    elif trust_fn_name == "confidence_trust":
        return confidence_trust
    elif trust_fn_name == "confidence_threshold_trust":
        return confidence_threshold_trust