import numpy as np
import pandas as pd


def rates_to_dataframe_temporal(rates):
    remove = []

    for key in rates.keys():
        if len(rates[key]) == 0:
            remove.append(key)

    for key in remove:
        del rates[key]

    return pd.DataFrame(rates)


def rates_to_dataframe_static(rates, *args):
    remove = []

    for key in rates.keys():
        if len(rates[key]) == 0:
            remove.append(key)

    for key in remove:
        del rates[key]

    return pd.DataFrame(rates)


def predictions_to_dataframe_static(predictions, *args):
    return pd.DataFrame(predictions)


def predictions_to_dataframe_temporal(predictions):
    return pd.DataFrame(predictions)


def get_rate_formatting_fn(temporal):
    if temporal:
        return rates_to_dataframe_temporal
    else:
        return rates_to_dataframe_static


def get_prediction_formatting_fn(temporal):
    if temporal:
        return predictions_to_dataframe_temporal
    else:
        return predictions_to_dataframe_static