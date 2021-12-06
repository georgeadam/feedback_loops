import numpy as np
import pandas as pd


def rates_to_dataframe_temporal(rates, train_year_limit, update_year_limit):
    data = {"rate": [], "rate_type": [], "year": [], "update_type": [], "seed": []}

    for update_type in rates.keys():
        for name in rates[update_type].keys():
            if name not in ["loss"]:
                for i in range(len(rates[update_type][name])):
                    data["rate"] += rates[update_type][name][i]
                    data["rate_type"] += [name] * (len(rates[update_type][name][i]))
                    data["year"] +=  list(np.arange(train_year_limit + 1, update_year_limit + 1))
                    data["update_type"] += [update_type] * (len(rates[update_type][name][i]))
                    data["seed"] += rates[update_type]["seed"][i]

    return pd.DataFrame(data)


def rates_to_dataframe_static(rates, *args):
    data = {"rate": [], "rate_type": [], "num_updates": [], "update_type": [], "seed": []}

    for update_type in rates.keys():
        for name in rates[update_type].keys():
            for i in range(len(rates[update_type][name])):
                if name not in ["loss"]:
                    data["rate"] += rates[update_type][name][i]
                    data["rate_type"] += [name] * len(rates[update_type][name][i])
                    data["num_updates"] +=  list(np.arange(len(rates[update_type][name][i])))
                    data["update_type"] += [update_type] * len(rates[update_type][name][i])
                    data["seed"] += rates[update_type]["seed"][i]

    return pd.DataFrame(data)


def get_result_formatting_fn(temporal):
    if temporal:
        return rates_to_dataframe_temporal
    else:
        return rates_to_dataframe_static