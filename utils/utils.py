import os
from river import metrics


def return_metrics():
    return {"accuracy": metrics.BalancedAccuracy(), "kappa": metrics.CohenKappa()}


def make_dir(path):
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)


def update_perf(perf, perf_values, predictions, cl_table, m):
    perf[m] = {"total": return_metrics(), "concept": return_metrics()}

    perf_values[m] = {
        "total": {"accuracy": [], "kappa": []},
        "concept": {"accuracy": [], "kappa": []},
    }
    perf_values["drifts"] = []

    predictions[m] = []

    cl_table[m] = {metric: [] for metric in ["accuracy", "kappa"]}

    return perf, perf_values, predictions, cl_table

