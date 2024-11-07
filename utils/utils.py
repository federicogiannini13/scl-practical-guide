import os
from river import metrics
import river.utils as ru

ROLLING_WINDOWS = [100, 500, 1000, 5000]


def return_metrics():
    return {"accuracy": metrics.BalancedAccuracy(), "kappa": metrics.CohenKappa()}


def return_rolling(window):
    return {
        "accuracy": ru.Rolling(metrics.BalancedAccuracy(), window_size=window),
        "kappa": ru.Rolling(metrics.CohenKappa(), window_size=window),
    }


def make_dir(path):
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)


def update_perf(perf, perf_values, predictions, cl_table, m, rolling_windows):
    perf[m] = {
        "total": return_metrics(),
        "concept": return_metrics(),
    }
    for window in rolling_windows:
        perf[m][f"rolling_{window}"] = return_rolling(window)
        perf[m][f"rolling_{window}_reset"] = return_rolling(window)

    perf_values[m] = {
        "total": {"accuracy": [], "kappa": []},
        "concept": {"accuracy": [], "kappa": []},
    }
    for window in rolling_windows:
        perf_values[m][f"rolling_{window}"] = {"accuracy": [], "kappa": []}
    perf_values["drifts"] = []

    predictions[m] = []

    cl_table[m] = {metric: [] for metric in ["accuracy", "kappa"]}

    return perf, perf_values, predictions, cl_table
