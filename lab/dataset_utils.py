import os
import itertools
import numpy as np
import pandas as pd
import pickle


def gen_conf(set1, set2):
    set2_perm = itertools.permutations(set2)
    all_couples = [[(x, y) for x, y in zip(set1, perm)] for perm in set2_perm]
    index = np.random.randint(0, len(all_couples))
    sel_couples = all_couples[index]
    np.random.shuffle(sel_couples)
    return sel_couples


def build_conf_df(root, dataset, df, experiences, conf, train="train"):
    dfs_conf = []
    for task, exp in enumerate(experiences):
        df_exp = df[df["target"].isin(exp)].reset_index(drop=True)
        df_exp["target"] = df_exp["task1"]
        df_exp = (
            df_exp.drop(columns=[f"task{i}" for i in range(1, 6)])
            .sample(frac=1)
            .reset_index(drop=True)
        )
        df_exp = df_exp.iloc[: len(df_exp) // 10 * 10]
        df_exp["task"] = task + 1
        dfs_conf.append(df_exp)
    df_conf = pd.concat(dfs_conf).reset_index(drop=True)
    df_conf.to_csv(
        os.path.join(root, "datasets", f"{dataset}_incremental_{conf}conf_{train}.csv"),
        index=False,
    )
    return df_conf


def make_exp_incremental_train(root, dataset, df, set1, set2, nconfs=10):
    try:
        with open(
            os.path.join(root, "datasets", f"{dataset}_incremental_confs.pkl"), "rb"
        ) as f:
            confs = pickle.load(f)
    except:
        confs = []
    count_confs = 0
    dfs = []
    while count_confs < nconfs:
        experiences = gen_conf(set1, set2)
        print(experiences, end=" ")
        if experiences in confs:
            print("REJECTED")
            continue
        print("ACCEPTED")
        confs.append(experiences)
        count_confs += 1
        df_conf = build_conf_df(root, dataset, df, experiences, len(confs), "train")
        dfs.append(df_conf)
    with open(
        os.path.join(root, "datasets", f"{dataset}_incremental_confs.pkl"), "wb"
    ) as f:
        pickle.dump(confs, f)
    return dfs


def make_exp_incremental_test(root, dataset, df):
    with open(
        os.path.join(root, "datasets", f"{dataset}_incremental_confs.pkl"), "rb"
    ) as f:
        confs = pickle.load(f)
    dfs = []
    for count, experiences in enumerate(confs):
        df_conf = build_conf_df(root, dataset, df, experiences, count + 1, "test")
        dfs.append(df_conf)
    return dfs


def get_conf_perm():
    perms = list(itertools.permutations([i for i in range(1, 6)]))
    index = np.random.randint(0, len(perms))
    return perms[index]


def build_conf_df_sml(
    root,
    dataset,
    df,
    conf,
    conf_number,
    train="train",
    n_split=5,
    tasks_perc=None,
    suffix="",
):
    if tasks_perc is not None:
        n_split = len(tasks_perc)
    dfs = [[] for _ in range(n_split)]
    for target in df["target"].unique():
        df_target = df[df["target"] == target].sample(frac=1)
        if tasks_perc is None:
            len_split = len(df_target) // n_split
            ranges = [
                (i * len_split, (i + 1) * len_split) for i in range(n_split - 1)
            ] + [((n_split - 1) * len_split, len(df_target))]
        else:
            ranges = [0] + [int(p*len(df_target)) for p in tasks_perc[:-1]]
            ranges = [ranges[i]+ranges[i+1] for i in range(len(ranges)-1)]
            ranges = (
                    [(0, ranges[0])] +
                    [(ranges[i], ranges[i+1]) for i in range(len(ranges)-1)] +
                      [(ranges[-1], len(df_target))]
            )
        for i, (start, end) in enumerate(ranges):
            dfs[i].append(df_target.iloc[start:end, :].reset_index(drop=True))

    final_dfs = [pd.concat(dfs_).sample(frac=1).reset_index(drop=True) for dfs_ in dfs]
    for i, task in enumerate(conf[:n_split]):
        final_dfs[i]["target"] = final_dfs[i][f"task{task}"]
        final_dfs[i]["task"] = i + 1
        final_dfs[i] = final_dfs[i].drop(columns=[f"task{j}" for j in range(1, 6)])
        final_dfs[i] = final_dfs[i].iloc[: len(final_dfs[i]) // 10 * 10, :]
    final_df = pd.concat(final_dfs).reset_index(drop=True)
    final_df.to_csv(
        os.path.join(
            root, "datasets", f"{dataset}_sml{suffix}_{conf_number}conf_{train}.csv"
        ),
        index=False,
    )
    return final_df


def make_exp_sml_train(root, dataset, df, nconfs=10, n_split=5, tasks_perc=None, suffix=""):
    try:
        with open(
            os.path.join(root, "datasets", f"{dataset}_sml{suffix}_confs.pkl"), "rb"
        ) as f:
            confs = pickle.load(f)
    except:
        confs = []
    count_conf = 0

    final_dfs = []
    if tasks_perc is not None:
        n_split = len(tasks_perc)
    while count_conf < nconfs:
        conf = get_conf_perm()
        print(conf[:n_split], end=" ")
        if conf in confs:
            print("REJECTED")
            continue
        print("ACCEPTED")
        confs.append(conf)
        count_conf += 1
        final_dfs.append(
            build_conf_df_sml(
                root,
                dataset,
                df,
                conf,
                len(confs),
                "train",
                n_split=n_split,
                tasks_perc=tasks_perc,
                suffix=suffix,
            )
        )
    with open(
        os.path.join(root, "datasets", f"{dataset}_sml{suffix}_confs.pkl"), "wb"
    ) as f:
        pickle.dump(confs, f)
    return final_dfs


def make_exp_sml_test(root, dataset, df, suffix="", n_split=5):
    with open(
        os.path.join(root, "datasets", f"{dataset}_sml{suffix}_confs.pkl"), "rb"
    ) as f:
        confs = pickle.load(f)
    dfs = []
    for count, experiences in enumerate(confs):
        df_conf = build_conf_df_sml(
            root,
            dataset,
            df,
            experiences,
            count + 1,
            "test",
            n_split=n_split,
            suffix=suffix,
        )
        dfs.append(df_conf)
    return dfs


def check_distr(exps):
    check_ok = True
    for i, e in enumerate(exps):
        for task in range(1, 6):
            df_task = e[e["task"] == task]
            vc = df_task["target"].value_counts()
            if len(vc) != 2:
                print(f"CONF {i + 1}, TASK: {task}")
                check_ok = False
    if check_ok:
        print("OK")
