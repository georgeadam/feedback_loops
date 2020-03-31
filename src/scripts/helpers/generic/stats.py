import numpy as np


def summarize_stats(stats):
    metrics = ["median", "mean", "std", "min", "max"]
    summary = {stage: {key: {} for key in stats[stage].keys()} for stage in stats.keys()}

    for stage in stats.keys():
        for key in stats[stage].keys():
            for metric in metrics:
                fn = getattr(np, metric)
                res = fn(stats[stage][key])

                if res.dtype.type == np.int64:
                    res = int(res)
                else:
                    res = float(res)
                summary[stage][key][metric] = res

    return summary