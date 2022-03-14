import os
import tqdm
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import wandb

from bo_protein.gfp_data import utils as gfp_utils
from bo_protein.gfp_data import transforms
from bo_protein.mavedb_data import utils as mavedb_utils
from bo_protein.stability_data import utils as stab_utils
from bo_protein.models import surrogates
from bo_protein.models.trainer import Trainer
from bo_protein import acquisition

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

START_AND_END_COUNT = 2


def ind_splits(source, task="Quantum Yield", split=0.9, train_wo_cutoff=False, seed=0):
    task = task if task else ""

    if source in ["localfl", "fpbase"]:
        train, test = gfp_utils.load_data(
            "./bo_protein", source, task, split, train_wo_cutoff, seed=seed
        )
    elif source in ["stability"]:
        train, test = stab_utils.load_data(
            "./bo_protein", source, task, split, train_wo_cutoff, seed=seed
        )
    else:
        train, test = mavedb_utils.load_data(
            "./bo_protein", source, split, train_wo_cutoff, seed=seed
        )

    X_train, Y_train = train
    X_test, Y_test = test
    data = (X_train, Y_train, X_test, Y_test)

    X = np.concatenate([X_train, X_test], axis=0)
    max_len = max([len(x) for x in X]) + START_AND_END_COUNT

    tag = f'{source}_{task.lower().replace(" ", "_")}'

    return data, max_len, tag


def ood_splits(source, task, train_wo_cutoff=False):
    task = task if task else ""

    if source in ["localfl", "fpbase"]:
        train, test = gfp_utils.load_data_mutation_split(
            "./gfp_bayesopt", source, task, train_wo_cutoff=train_wo_cutoff
        )
    elif source in ["stability"]:
        train, test = stab_utils.load_data_mutation_split(
            "./gfp_bayesopt", source, task, train_wo_cutoff=train_wo_cutoff
        )
    else:
        train, test = mavedb_utils.load_data_mutation_split(
            "./gfp_bayesopt", source, train_wo_cutoff=train_wo_cutoff
        )

    X_train, Y_train, mutations_train = train
    X_test, Y_test, mutations_test = test
    data = (X_train, Y_train, X_test, Y_test)

    X = np.concatenate([X_train, X_test], axis=0)
    max_len = np.max([len(x) for x in X]) + START_AND_END_COUNT

    tag = f'{source}_{task.lower().replace(" ", "_")}_ood'

    return data, max_len, tag


def train_and_evaluate(config, data, max_len, tag):
    X_train, Y_train, X_test, Y_test = data

    (
        data_finetune,
        _,
        _,
    ) = ind_splits("localfl", None, split=1.0)
    X_finetune = data_finetune[0]

    X_train = X_train[:2500]
    Y_train = Y_train[:2500]

    X_test = X_test[:100]
    Y_test = Y_test[:100]

    # surrogate = surrogates.DeepEnsemble(config).fit(X_train, Y_train, X_test, Y_test, log_prefix=tag)
    # surrogate = surrogates.TransformerBootstrapSVMAcquisition(config).finetune(X_finetune, steps=100)
    # surrogate = surrogates.BootstrapXGBoostAcquisition(config).fit(X_train, Y_train, X_test, Y_test, log_prefix=tag)
    # surrogate = surrogates.EmbeddingGP(config)
    surrogate = surrogates.SSKGP({"max_len": max_len})
    surrogate.fit(X_train, Y_train, X_test, Y_test, log_prefix=tag)

    bs = config.get("bs", 200)
    metrics = surrogate.evaluate(X_test, Y_test, bs=bs, log_prefix=tag)

    print(metrics)

    return surrogate, {tag: metrics}


def evaluate_regression(config):
    train_and_evaluate(config, *ind_splits("stability", None))


def main(**config):
    config.setdefault("num_trials", 3)
    config.setdefault("ensemble_size", 5)
    config.setdefault("num_epochs", 400)
    config.setdefault("bootstrap_ratio", None)
    config.setdefault("weight_decay", 0.0)
    config.setdefault("holdout_ratio", 0.05)
    config.setdefault("early_stopping", True)
    config.setdefault("lr", 3e-4)
    config.setdefault("max_shift", 4)
    config.setdefault("mask_size", 2)

    wandb.init(project="gfp-regression-nb", config=config)

    evaluate_regression(config)


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", default="dryrun")

    from fire import Fire

    Fire(main)
