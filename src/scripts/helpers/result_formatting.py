import numpy as np
import pandas as pd

from src.utils.data import TEMPORAL_DATA_TYPES


def results_to_dataframe_temporal(rates, train_year_limit, update_year_limit):
    data = {"rate": [], "rate_type": [], "year": [], "update_type": []}

    for update_type in rates.keys():
        for name in rates[update_type].keys():
            if name != "loss":
                for i in range(len(rates[update_type][name])):
                    data["rate"] += rates[update_type][name][i]
                    data["rate_type"] += [name] * (len(rates[update_type][name][i]))
                    data["year"] +=  list(np.arange(train_year_limit, update_year_limit))
                    data["update_type"] += [update_type] * (len(rates[update_type][name][i]))

    return pd.DataFrame(data)


def results_to_dataframe_static(rates, *args):
    data = {"rate": [], "rate_type": [], "num_updates": [], "update_type": []}

    for update_type in rates.keys():
        for name in rates[update_type].keys():
            for i in range(len(rates[update_type][name])):
                data["rate"] += rates[update_type][name][i]
                data["rate_type"] += [name] * len(rates[update_type][name][i])
                data["num_updates"] +=  list(np.arange(len(rates[update_type][name][i])))
                data["update_type"] += [update_type] * len(rates[update_type][name][i])

    return pd.DataFrame(data)


def get_result_formatting_fn(temporal):
    if temporal:
        return results_to_dataframe_temporal
    else:
        return results_to_dataframe_static