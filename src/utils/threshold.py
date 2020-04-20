import numpy as np

from src.utils.metrics import compute_all_rates


def find_threshold(y: np.ndarray, y_prob: np.ndarray, desired_rate: str, desired_value: float,
                   tol: float=0.01) -> float:
    thresholds = np.linspace(0.01, 0.999, 1000)
    best_threshold = None
    best_diff = float("inf")

    l = 0
    r = len(thresholds)

    while l < r:
        mid = l + (r - l) // 2
        threshold = thresholds[mid]

        if y_prob.shape[1] > 1:
            temp_pred = y_prob[:, 1] >= threshold
        else:
            temp_pred = y_prob[:, 0] >= threshold

        rates = compute_all_rates(y, temp_pred, y_prob)

        direction = get_direction(desired_rate)

        if abs(rates[desired_rate] - desired_value) <= desired_value * tol:
            return threshold
        elif (rates[desired_rate] < desired_value and direction == "decreasing") or (rates[desired_rate] > desired_value and direction == "increasing"):
            l = mid + 1
        else:
            r = mid - 1

        if abs(rates[desired_rate] - desired_value) <= best_diff:
            best_diff = abs(rates[desired_rate] - desired_value)
            best_threshold = threshold


    return best_threshold


def get_direction(rate: str) -> str:
    if rate == "fpr" or rate == "tnr":
        return "increasing"
    elif rate == "fnr" or rate == "tpr":
        return "decreasing"