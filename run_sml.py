DATASETS = [
    f"mnist_red30_incremental_{c}conf"
    for c in range(1,11)
]
ROOT = "/Users/federicogiannini/Library/CloudStorage/OneDrive-PolitecnicodiMilano/SML_CL"
SUFFIX = ""

import pickle
import pandas as pd
from river import forest, stream
from river import tree
from utils.sml_utils import test_cl
import os
from utils.utils import return_metrics, make_dir

if SUFFIX is not None or SUFFIX != "":
    SUFFIX = "_" + SUFFIX
for DATASET in DATASETS:
    df = pd.read_csv(os.path.join(ROOT, "datasets", f"{DATASET}_train.csv"), nrows=1)
    last_task = df["task"].iloc[0]
    converters = {c: float for c in df.columns if "feat" in c}
    converters["target"] = int
    converters["task"] = int

    data_stream = stream.iter_csv(
        os.path.join(ROOT, "datasets", f"{DATASET}_train.csv"),
        converters=converters,
        target="target"
    )

    models = {
        "arf": forest.ARFClassifier(leaf_prediction="nb"),
        "hat": tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=100,
            delta=1e-5,
            leaf_prediction="nb",
            nb_threshold=10,
        )
    }

    perf = {
        m: {
            "total": return_metrics(),
            "concept": return_metrics()
        } for m in models
    }

    perf_values = {
        m: {
            "total": {"accuracy": [], "kappa": []},
            "concept": {"accuracy": [], "kappa": []}
        } for m in models
    }
    perf_values["drifts"] = []

    predictions = {m:[] for m in models}


    cl_table = {
        m: {
            metric: [] for metric in ["accuracy", "kappa"]
        } for m in models
    }

    df_test = pd.read_csv(os.path.join(ROOT, "datasets", f"{DATASET}_test.csv"))

    X_test = []
    y_test = []
    for task in df_test["task"].unique():
        df_task = df_test[df_test["task"]==task]
        X_test.append(df_task.iloc[:,:-2].values)
        y_test.append(list(df_task["target"]))


    df = pd.read_csv(os.path.join(ROOT, "datasets", f"{DATASET}_train.csv"))


    for idx, (x, y) in enumerate(data_stream):
        print(f"{DATASET} Prequential {idx+1}", end="\r")
        if last_task != x["task"]:
            print()
            print(f"DRIFT {idx+1}")
            for m in perf:
                perf[m]["concept"] = return_metrics()
            cl_table = test_cl(cl_table, models, X_test, y_test)
            perf_values["drifts"].append(idx)
        last_task = x["task"]
        del x["task"]
        for m in models:
            pred = models[m].predict_one(x)
            pred = 0 if pred is None else pred
            predictions[m].append(pred)
            for method in ["total", "concept"]:
                for metric in perf[m][method]:
                    perf[m][method][metric].update(y, pred)
                    perf_values[m][method][metric].append(perf[m][method][metric].get())
            models[m].learn_one(x, y)
    cl_table = test_cl(cl_table, models, X_test, y_test)

    make_dir(os.path.join(ROOT, "performance", DATASET))
    with open(os.path.join(ROOT, "performance", DATASET, f"performance_sml{SUFFIX}.pkl"), "wb") as f:
        pickle.dump(perf_values, f)
    with open(os.path.join(ROOT, "performance", DATASET, f"cl_table_sml{SUFFIX}.pkl"), "wb") as f:
        pickle.dump(cl_table, f)