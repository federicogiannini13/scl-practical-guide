from sklearn import metrics as skm

from utils.cl_utils.custom_mlp import CustomMLP
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import bwt_metrics, accuracy_metrics
import pickle
import os
import numpy as np
from avalanche.training.plugins import ReplayPlugin
from utils.cl_utils.strategies.agem import AGEMPlugin
from utils.cl_utils.strategies.ewc import EWCPlugin
from utils.cl_utils.strategies.lwf import LwFPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.supervised import Naive

from utils.utils import return_metrics, make_dir


def return_components(input_size=30):
    model = CustomMLP(input_size=input_size)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "eval_plugin": eval_plugin,
    }


def run_strategy(
    cl_strategy,
    model,
    strategy,
    root,
    dataset,
    online_train_stream,
    X_test,
    y_test,
    perf,
    perf_values,
    predictions,
    cl_table,
):
    last_task = 0
    idx = 0
    for experience in online_train_stream:
        for _, _, task in experience.dataset:
            break
        if last_task != task:
            perf_values["drifts"].append(idx)
            cl_table = test_cl(cl_table, model, strategy, X_test, y_test)
            perf[strategy]["concept"] = return_metrics()
        last_task = task
        print(
            dataset,
            strategy,
            "Minibatch:",
            experience.current_experience,
            ", Task:",
            task,
            ", Size:",
            len(experience.dataset),
            end="\r",
        )
        for x, y, _ in experience.dataset:
            y_hat = np.argmax(model(x).detach().numpy())
            predictions[strategy].append(y_hat)
            for metric in ("accuracy", "kappa"):
                for eval_ in ("total", "concept"):
                    perf[strategy][eval_][metric].update(y, y_hat)
                    perf_values[strategy][eval_][metric].append(
                        perf[strategy][eval_][metric].get()
                    )
        cl_strategy.train(experience, eval_streams=[])
        idx += 1
    cl_table = test_cl(cl_table, model, strategy, X_test, y_test)

    make_dir(os.path.join(root, "performance", dataset))
    with open(
        os.path.join(root, "performance", dataset, f"performance_cl.pkl"), "wb"
    ) as f:
        pickle.dump(perf_values, f)
    with open(
        os.path.join(root, "performance", dataset, f"cl_table_cl.pkl"), "wb"
    ) as f:
        pickle.dump(cl_table, f)
    with open(
        os.path.join(root, "performance", dataset, f"predictions_cl.pkl"), "wb"
    ) as f:
        pickle.dump(predictions, f)

    return perf, perf_values, predictions, cl_table


def extract_kwargs(extract, kwargs):
    """
    checks and extracts
    the arguments
    listed in extract
    """
    init_dict = {}
    for word in extract:
        if word not in kwargs:
            raise AttributeError(f"Missing attribute {word} in provided configuration")
        init_dict.update({word: kwargs[word]})
    return init_dict


def create_strategy(
    name: str,
    components: dict,
    mb_size,
    strategy_kwargs=None,
):
    plugins = []

    if name == "naive":
        return Naive(
            model=components["model"],
            optimizer=components["optimizer"],
            criterion=components["criterion"],
            train_mb_size=mb_size,
            train_epochs=1,
            evaluator=components["eval_plugin"],
            eval_mb_size=mb_size,
        )

    elif name == "er":
        specific_args = extract_kwargs(["mem_size", "batch_size_mem"], strategy_kwargs)
        storage_policy = ClassBalancedBuffer(
            max_size=specific_args["mem_size"], adaptive_size=True
        )
        replay_plugin = ReplayPlugin(**specific_args, storage_policy=storage_policy)
        plugins.append(replay_plugin)

    elif name == "lwf":
        specific_args_lwf = extract_kwargs(["alpha", "temperature"], strategy_kwargs)
        lwf_plugin = LwFPlugin(**specific_args_lwf)
        plugins.append(lwf_plugin)

    elif name == "ewc":
        ewc_plugin = EWCPlugin(
            ewc_lambda=0.4,
        )
        plugins.append(ewc_plugin)

    elif name == "er_lwf":
        specific_args_replay = extract_kwargs(
            ["mem_size", "batch_size_mem"], strategy_kwargs
        )
        specific_args_lwf = extract_kwargs(["alpha", "temperature"], strategy_kwargs)
        storage_policy = ClassBalancedBuffer(
            max_size=specific_args_replay["mem_size"], adaptive_size=True
        )
        replay_plugin = ReplayPlugin(
            **specific_args_replay, storage_policy=storage_policy
        )
        lwf_plugin = LwFPlugin(**specific_args_lwf)
        plugins.append(replay_plugin)
        plugins.append(lwf_plugin)

    elif name == "agem":
        specific_args = extract_kwargs(
            ["mem_size", "sample_size"],
            strategy_kwargs,
        )
        agem_plugin = AGEMPlugin(**specific_args)
        plugins.append(agem_plugin)

    return SupervisedTemplate(
        model=components["model"],
        optimizer=components["optimizer"],
        criterion=components["criterion"],
        train_mb_size=mb_size,
        eval_mb_size=mb_size,
        train_epochs=1,
        evaluator=components["eval_plugin"],
        plugins=plugins,
    )


def test_cl(cl_table, model, strategy, X_test, y_test):
    print()
    print("Test evaluation")
    for metric in ["accuracy", "kappa"]:
        cl_table[strategy][metric].append([])
    for X_test_task, y_test_task in zip(X_test, y_test):
        pred = model(X_test_task).detach().numpy().argmax(axis=1)
        cl_table[strategy]["accuracy"][-1].append(skm.accuracy_score(y_test_task, pred))
        cl_table[strategy]["kappa"][-1].append(skm.cohen_kappa_score(y_test_task, pred))
    return cl_table
