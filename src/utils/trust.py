from typing import Any


def full_trust(*args: Any, **kwargs: Any) -> float:
    return 1.0


def conditional_trust(model_fpr: float=0.0, **kwargs: Any) -> float:
    return 1 - model_fpr


def constant_trust(clinician_trust: float=1.0, **kwargs: Any) -> float:
    return clinician_trust
