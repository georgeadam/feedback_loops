from scipy.stats import ttest_ind
import numpy as np


def detect_feedback_loop(y_train, y_update):
    t, p = ttest_ind(y_train, y_update)

    return p


def check_feedback_loop(y_train, agg_y_update, rates):
    p = detect_feedback_loop(y_train, agg_y_update)

    rates["detection"][-1] = p