import numpy as np
import pandas as pd

from typing import Any, Callable, Dict

def results_to_dataframe_conditional_temporal(rates: Dict, train_year_limit: int, update_year_limit: int) -> pd.DataFrame:
    data = {"rate": [], "rate_type": [], "year": [], "model_fpr": [], "clinician_fpr": [], "update_type": []}

    for update_type in rates.keys():
        for model_fpr in rates[update_type].keys():
            for clinician_fpr in rates[update_type][model_fpr].keys():
                for rate_type in rates[update_type][model_fpr][clinician_fpr].keys():
                    if rate_type != "loss":
                        for i in range(len(rates[update_type][model_fpr][clinician_fpr][rate_type])):
                            data["rate"] += rates[update_type][model_fpr][clinician_fpr][rate_type][i]
                            data["rate_type"] += [rate_type] * (len(rates[update_type][model_fpr][clinician_fpr][rate_type][i]))
                            data["year"] +=  list(np.arange(train_year_limit + 1, update_year_limit + 1))
                            data["model_fpr"] += [model_fpr] * len(rates[update_type][model_fpr][clinician_fpr][rate_type][i])
                            data["clinician_fpr"] += [clinician_fpr] * len(rates[update_type][model_fpr][clinician_fpr][rate_type][i])
                            data["update_type"] += [update_type] * len(rates[update_type][model_fpr][clinician_fpr][rate_type][i])

    return pd.DataFrame(data)


def results_to_dataframe_conditional_static(rates: Dict, *args: Any) -> pd.DataFrame:
    data = {"rate": [], "rate_type": [], "num_updates": [], "model_fpr": [], "clinician_fpr": [], "update_type": []}

    for update_type in rates.keys():
        for model_fpr in rates[update_type].keys():
            for clinician_fpr in rates[update_type][model_fpr].keys():
                for rate_type in rates[update_type][model_fpr][clinician_fpr].keys():
                    for i in range(len(rates[update_type][model_fpr][clinician_fpr][rate_type])):
                        data["rate"] += rates[update_type][model_fpr][clinician_fpr][rate_type][i]
                        data["rate_type"] += [rate_type] * len(rates[update_type][model_fpr][clinician_fpr][rate_type][i])
                        data["num_updates"] +=  list(np.arange(len(rates[update_type][model_fpr][clinician_fpr][rate_type][i])))
                        data["model_fpr"] += [model_fpr] * len(rates[update_type][model_fpr][clinician_fpr][rate_type][i])
                        data["clinician_fpr"] += [clinician_fpr] * len(rates[update_type][model_fpr][clinician_fpr][rate_type][i])
                        data["update_type"] += [update_type] * len(rates[update_type][model_fpr][clinician_fpr][rate_type][i])

    return pd.DataFrame(data)


def results_to_dataframe_constant_temporal(rates: Dict, train_year_limit: int, update_year_limit: int) -> pd.DataFrame:
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


def results_to_dataframe_constant_static(rates: Dict, *args: Any) -> pd.DataFrame:
    data = {"rate": [], "rate_type": [], "num_updates": [], "trust": []}

    for trust in rates.keys():
        for rate_type in rates[trust].keys():
            for i in range(len(rates[trust][rate_type])):
                data["rate"] += rates[trust][rate_type][i]
                data["rate_type"] += [rate_type] * len(rates[trust][rate_type][i])
                data["num_updates"] +=  list(np.arange(len(rates[trust][rate_type][i])))
                data["trust"] += [trust] * len(rates[trust][rate_type][i])

    return pd.DataFrame(data)


def get_result_formatting_fn(temporal: bool, trust_type: str) -> Callable[[Dict, Any], pd.DataFrame]:
    if temporal and (trust_type == "conditional" or trust_type == "confidence"):
        return results_to_dataframe_conditional_temporal
    elif temporal and trust_type == "constant":
        return results_to_dataframe_constant_temporal
    elif not temporal and (trust_type == "conditional" or trust_type == "confidence"):
        return results_to_dataframe_conditional_static
    else:
        return results_to_dataframe_constant_static