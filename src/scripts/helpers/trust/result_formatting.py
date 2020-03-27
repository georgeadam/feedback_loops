import numpy as np
import pandas as pd


def results_to_dataframe_conditional_temporal(rates, train_year_limit, update_year_limit):
    data = {"rate": [], "rate_type": [], "year": [], "model_fpr": [], "clinician_fpr": []}

    for model_fpr in rates.keys():
        for clinician_fpr in rates[model_fpr].keys():
            for rate_type in rates[model_fpr][clinician_fpr].keys():
                if rate_type != "loss":
                    for i in range(len(rates[model_fpr][clinician_fpr][rate_type])):
                        data["rate"] += rates[model_fpr][clinician_fpr][rate_type][i]
                        data["rate_type"] += [rate_type] * (len(rates[model_fpr][clinician_fpr][rate_type][i]))
                        data["year"] +=  list(np.arange(train_year_limit + 1, update_year_limit + 1))
                        data["model_fpr"] += [model_fpr] * len(rates[model_fpr][clinician_fpr][rate_type][i])
                        data["clinician_fpr"] += [clinician_fpr] * len(rates[model_fpr][clinician_fpr][rate_type][i])

    return pd.DataFrame(data)


def results_to_dataframe_conditional_static(rates, *args):
    data = {"rate": [], "rate_type": [], "num_updates": [], "model_fpr": [], "clinician_fpr": []}

    for model_fpr in rates.keys():
        for clinician_fpr in rates[model_fpr].keys():
            for rate_type in rates[model_fpr][clinician_fpr].keys():
                for i in range(len(rates[model_fpr][clinician_fpr][rate_type])):
                    data["rate"] += rates[model_fpr][clinician_fpr][rate_type][i]
                    data["rate_type"] += [rate_type] * len(rates[model_fpr][clinician_fpr][rate_type][i])
                    data["num_updates"] +=  list(np.arange(len(rates[model_fpr][clinician_fpr][rate_type][i])))
                    data["model_fpr"] += [model_fpr] * len(rates[model_fpr][clinician_fpr][rate_type][i])
                    data["clinician_fpr"] += [clinician_fpr] * len(rates[model_fpr][clinician_fpr][rate_type][i])

    return pd.DataFrame(data)


def results_to_dataframe_constant_temporal(rates, train_year_limit, update_year_limit):
    data = {"rate": [], "rate_type": [], "year": [], "trust": []}

    for trust in rates.keys():
        for rate_type in rates[trust].keys():
            if rate_type != "loss":
                for i in range(len(rates[trust][rate_type])):
                    data["rate"] += rates[trust][rate_type][i]
                    data["rate_type"] += [rate_type] * (len(rates[trust][rate_type][i]))
                    data["year"] +=  list(np.arange(train_year_limit + 1, update_year_limit + 1))
                    data["trust"] += [trust] * len(rates[trust][rate_type][i])

    return pd.DataFrame(data)


def results_to_dataframe_constant_static(rates, *args):
    data = {"rate": [], "rate_type": [], "num_updates": [], "trust": []}

    for trust in rates.keys():
        for rate_type in rates[trust].keys():
            for i in range(len(rates[trust][rate_type])):
                data["rate"] += rates[trust][rate_type][i]
                data["rate_type"] += [rate_type] * len(rates[trust][rate_type][i])
                data["num_updates"] +=  list(np.arange(len(rates[trust][rate_type][i])))
                data["trust"] += [trust] * len(rates[trust][rate_type][i])

    return pd.DataFrame(data)


def get_result_formatting_fn(temporal, trust_type):
    if temporal and trust_type == "conditional":
        return results_to_dataframe_conditional_temporal
    elif temporal and trust_type == "constant":
        return results_to_dataframe_constant_temporal
    elif not temporal and trust_type == "conditional":
        return results_to_dataframe_conditional_static
    else:
        return results_to_dataframe_constant_static