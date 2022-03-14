import os
import numpy as np
import pandas as pd
from Levenshtein import distance

avGFP = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"


def dist_to_avGFP(seq):
    return distance(avGFP, seq)


def _load_fpbase_data(root, task, cutoff_dist):
    file_path = os.path.join(root, "gfp_data/fpbase_data/fpbase_sequences.csv")
    df = pd.read_csv(file_path, index_col=0)
    df = df[df[task].notnull()]
    df = df[df["num_mutations"] < cutoff_dist]
    df["Seq"] = df["Seq"].str[1:]
    X, Y = df["Seq"].to_numpy(), df[task].to_numpy()
    mutations = df["num_mutations"].to_numpy()
    return X, Y, mutations


def _load_localfl_data(root, task, cutoff_dist):
    file_path = os.path.join(root, "gfp_data/localfl_data/fluorescence_test.json")
    with open(file_path, "r") as fd:
        test_df = pd.read_json(fd)

    file_path = os.path.join(root, "gfp_data/localfl_data/fluorescence_train.json")
    with open(file_path, "r") as fd:
        train_df = pd.read_json(fd)

    file_path = os.path.join(root, "gfp_data/localfl_data/fluorescence_valid.json")
    with open(file_path, "r") as fd:
        valid_df = pd.read_json(fd)

    df = pd.concat([test_df, train_df, valid_df])
    df = df[df["num_mutations"] < cutoff_dist]
    df["log_fluorescence"] = df["log_fluorescence"].apply(lambda x: x[0])
    X, Y = df["primary"].to_numpy(), df["log_fluorescence"].to_numpy()
    mutations = df["num_mutations"].to_numpy()
    return X, Y, mutations

def _load_bighat_data(root, task, cutoff_dist):
    file_path = os.path.join(root, "gfp_data/bighat_data/gfp_wetlab_results_round_1_base_seq.csv")
    df = pd.read_csv(file_path)
    X, Y = df["base_seq"].to_numpy(), df[task].to_numpy()

    file_path = os.path.join(root, "gfp_data/bighat_data/gfp_wetlab_results_round_1_opt_seq.csv")
    df = pd.read_csv(file_path)
    X = np.concatenate([df.opt_seq.to_numpy(), X])
    Y = np.concatenate([df[task].to_numpy(), Y])

    return X, Y

def _load_data(root, source, task, cutoff_dist=20):
    if source == "fpbase":
        return _load_fpbase_data(root, task, cutoff_dist)
    elif source == "localfl":
        return _load_localfl_data(root, task, cutoff_dist)
    elif source == "bighat":
        return _load_bighat_data(root, task, cutoff_dist)
    

def _normalize(Y):
    Y_mean, Y_std = Y.mean(), Y.std()
    Y = (Y - Y_mean) / Y_std
    return Y


def _shuffle(X, Y, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
    return X, Y, idx


def _divide_by_mutations(X, Y, mutations, lower_perc=0.5, upper_perc=0.5):
    mutations_upper = np.quantile(mutations, lower_perc)
    mutations_lower = np.quantile(mutations, 1 - upper_perc)

    train_mask = mutations <= mutations_upper
    test_mask = mutations > mutations_lower

    X, Y, idx = _shuffle(X, Y)

    mutations = mutations[idx]
    train_mask = train_mask[idx]
    test_mask = test_mask[idx]

    return (X[train_mask], Y[train_mask], mutations[train_mask]), (
        X[test_mask],
        Y[test_mask],
        mutations[test_mask],
    )


def load_data(root, source, task, split=0.9, train_wo_cutoff=False, seed=0):
    X, Y, _ = _load_data(root, source, task)
    X, Y, _ = _shuffle(X, Y, seed=seed)

    num_train = int(split * len(X))
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_test, Y_test = X[num_train:], Y[num_train:]

    if train_wo_cutoff:
        X, Y, _ = _load_data(source, task, cutoff_dist=float("inf"))
        X_train = np.concatenate([X_train, X])
        Y_train = np.concatenate([Y_train, Y])

    Y = _normalize(np.concatenate([Y_train, Y_test]))
    Y_train, Y_test = Y[: len(Y_train)], Y[len(Y_train) :]

    return (X_train, Y_train), (X_test, Y_test)


def load_data_mutation_split(
    root, source, task, lower_perc=0.5, upper_perc=0.5, train_wo_cutoff=False
):
    assert (lower_perc + upper_perc) <= 1.0
    X, Y, mutations = _load_data(root, source, task)
    splits = _divide_by_mutations(X, Y, mutations)

    if train_wo_cutoff:
        X, Y, mutations = _load_data(source, task, cutoff_dist=float("inf"))

        train, test = splits
        X_train, Y_train, mutations_train = train

        X_train = np.concatenate([X_train, X])
        Y_train = np.concatenate([Y_train, Y])
        mutations = np.concatenate([mutations_train, mutations])

        train = (X_train, Y_train, mutations)
        splits = (train, test)

    (X_train, Y_train, mutations_train), (X_test, Y_test, mutations_test) = splits
    Y = _normalize(np.concatenate([Y_train, Y_test]))
    Y_train, Y_test = Y[: len(Y_train)], Y[len(Y_train) :]
    splits = (X_train, Y_train, mutations_train), (X_test, Y_test, mutations_test)

    return splits
