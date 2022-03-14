import wandb
import os
from Levenshtein import distance as edit_distance
import numpy as np
import pandas as pd
import time
from datetime import datetime
import copy
import math

from tqdm import tqdm

from regression import ind_splits

import bo_protein.utils
from bo_protein.optimizers.genetic import LocalGeneticOptimizer
from bo_protein.models import surrogates
from bo_protein import acquisition
from bo_protein.gfp_data import utils
from bo_protein.utils import mutation_list


def _generate_proposals(dataset_config, proposal_configs, task):
    print(os.getcwd())
    train_data, test_data, label_stats = make_dataset(dataset_config)
    label_mean, label_std = label_stats
    exclude_seqs = np.unique(np.concatenate([train_data[0], test_data[0]]))
    max_len = max([len(x) for x in exclude_seqs]) + 2
    # tag = f'{source}_{task.lower().replace(" ", "_")}_ind'

    proposal_batch = {
        "base_seq": [],
        "base_mutations": None,
        "opt_seq": [],
        "opt_mutations": None,
        "obj_val": [],
        "pred_base_label": [],
        "pred_opt_label": [],
        "acq_type": [],
        "task": task,
    }
    for prop_cfg in proposal_configs:
        num_proposals = prop_cfg.get('num_proposals', 0)
        if num_proposals < 1:
            continue
        else:
            print(f"[{prop_cfg['tag']}] generating {num_proposals} proposal sequence(s)")

        # construct surrogate
        surrogate_constr = prop_cfg.get('surrogate_constr', None)
        surrogate_config = prop_cfg.get('surrogate_config', None)
        surrogate_args = [] if surrogate_config is None else [surrogate_config]
        surrogate = None if surrogate_constr is None else surrogate_constr(*surrogate_args)
        if hasattr(surrogate, 'fit'):
            num_params = count_params(surrogate)
            num_train = train_data[0].shape[0]
            print(f"==== training surrogate w/ {num_params:.2e} params on {num_train} examples ====")
            surrogate.fit(*train_data, *test_data, log_prefix=prop_cfg.get('tag', ''))

        # construct acquisition
        acq_constr = prop_cfg.get('acq_constr', None)
        acq_config = prop_cfg.get('acq_config', None)
        acq_args = [obj for obj in [surrogate, acq_config] if obj is not None]
        acq_fn = None if acq_constr is None else acq_constr(*acq_args)

        optimizer_constr = prop_cfg.get('optimizer_constr', None)
        optimizer_kwargs = prop_cfg.get('optimizer_kwargs', None)
        optimizer_kwargs.update(dict(max_len=max_len - 2))  # subtract off start/end tokens
        optimizer = optimizer_constr(**optimizer_kwargs)

        print("==== optimizing proposals ====")
        optimizer_tag = f"{prop_cfg['tag']}/{task}/optimizer"
        start_pool = get_start_pool()
        avg_gfp_dist = [edit_distance(seq, utils.avGFP) for seq in start_pool]
        print(f"[{optimizer_tag}] start pool ({start_pool.shape[0]} sequences) info:")
        print(f"[{optimizer_tag}] max edit distance from avg. GFP: {max(avg_gfp_dist)}")
        print(
            f"[{optimizer_tag}] avg. edit distance from avg. GFP: {sum(avg_gfp_dist) / len(avg_gfp_dist):0.4f}"
        )

        pool_size = start_pool.shape[0]
        if pool_size < num_proposals and prop_cfg.get('unique_base_seq', True):
            raise RuntimeError(f'[{optimizer_tag}] {pool_size} unique base sequences, {num_proposals} required')

        opt_seqs, acq_vals, base_seqs = [], [], []
        for _ in tqdm(range(10 * num_proposals)):
            if len(opt_seqs) == num_proposals:
                break

            max_seq, max_reward, max_ancestor = optimizer.optimize(
                start_pool, acq_fn, top_k=1, log_prefix=optimizer_tag
            )

            if np.any(exclude_seqs == max_seq[0]):  # filter existing sequences
                print('winning sequence has already been proposed, repeating search')
                continue
            else:
                opt_seqs.append(max_seq[0])
                acq_vals.append(max_reward[0])
                base_seqs.append(max_ancestor[0])
                start_pool = start_pool[~(start_pool == max_ancestor[0])]
                exclude_seqs = np.concatenate([exclude_seqs, max_seq])

        opt_seqs = np.array(opt_seqs)
        acq_vals = np.array(acq_vals)
        base_seqs = np.array(base_seqs)
        opt_source = np.array([prop_cfg['tag']] * num_proposals)
        opt_seqs, acq_vals, base_seqs = postprocess_proposals(num_proposals, opt_seqs, base_seqs, acq_vals)
        proposal_info(opt_seqs, acq_vals)

        # get predicted labels, unnormalize
        acq_fn.acq_type = "posterior-mean"
        opt_pred_labels = acq_fn.score(opt_seqs)
        base_pred_labels = acq_fn.score(base_seqs)
        opt_pred_labels = opt_pred_labels * label_std + label_mean
        base_pred_labels = base_pred_labels * label_std + label_mean

        # add sequences to proposal batch
        proposal_batch["base_seq"].append(base_seqs)
        proposal_batch['opt_seq'].append(opt_seqs)
        proposal_batch['obj_val'].append(acq_vals)
        proposal_batch['pred_base_label'].append(base_pred_labels)
        proposal_batch['pred_opt_label'].append(opt_pred_labels)
        proposal_batch['acq_type'].append(opt_source)

    print("==== saving results ====")
    proposal_batch.update({
        key: np.concatenate(val) for key, val in proposal_batch.items() if isinstance(val, list) and isinstance(val[0], np.ndarray)
    })
    proposal_batch['base_mutations'] = np.array([mutation_list(utils.avGFP, seq) for seq in proposal_batch['base_seq']])
    proposal_batch['opt_mutations'] = np.array([mutation_list(anc, seq) for anc, seq in zip(proposal_batch['base_seq'], proposal_batch['opt_seq'])])

    # save results to .csv
    df = pd.DataFrame.from_dict(proposal_batch)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists("proposals"):
        os.mkdir("proposals")
    df.to_csv(f"proposals/gfp_batch_{dt_string}.csv", index=None)
    time.sleep(1)


def postprocess_proposals(
    num_proposals, raw_proposals, raw_ancestors, raw_rewards=None
):
    # filter duplicates
    _, unique_idx = np.unique(raw_proposals, return_index=True)
    proposals = raw_proposals[unique_idx]
    proposal_rewards = raw_rewards[unique_idx] if raw_rewards is not None else None
    ancestors = raw_ancestors[unique_idx]

    # sort and return top results
    num_proposals = min(proposals.shape[0], num_proposals)
    sorted_idx = (
        np.argsort(-proposal_rewards)
        if raw_rewards is not None
        else np.arange(proposals.shape[0])
    )
    proposals = proposals[sorted_idx][:num_proposals]
    rewards = (
        proposal_rewards[sorted_idx][:num_proposals]
        if raw_rewards is not None
        else None
    )
    ancestors = ancestors[sorted_idx][:num_proposals]

    return proposals, rewards, ancestors


def proposal_info(proposals, proposal_rewards):
    print(f"==== {proposals.shape[0]} Proposals ====")
    for seq, reward in zip(proposals, proposal_rewards):
        print(f"\t[Score {reward:0.4f}] {seq}")
    min_dist, max_dist = get_dist_stats(proposals)
    print(f"pairwise edit distance between proposals: min={min_dist}, max={max_dist}")

    avg_gfp_dist = [edit_distance(seq, utils.avGFP) for seq in proposals]
    print(f"max edit distance from avg. GFP: {max(avg_gfp_dist)}")
    print(
        f"avg. edit distance from avg. GFP: {sum(avg_gfp_dist) / len(avg_gfp_dist):0.4f}"
    )


def get_dist_stats(proposals):
    min_dist = float("inf")
    max_dist = 0
    for i in range(proposals.shape[0]):
        for j in range(i + 1, proposals.shape[0]):
            dist = edit_distance(proposals[i], proposals[j])
            min_dist = min(min_dist, dist)
            max_dist = max(max_dist, dist)
    return min_dist, max_dist


def get_start_pool():
    data, _, _ = ind_splits(
        "fpbase", "Brightness", split=0.95, train_wo_cutoff=False
    )
    base_seqs = np.concatenate([data[0], data[2]])
    base_targets = np.concatenate([data[1], data[3]])
    brightness_cutoff = np.quantile(base_targets, 0.5)
    base_seqs = base_seqs[base_targets > brightness_cutoff]

    bighat_data = utils._load_bighat_data('./bo_protein', 'MaxRFU', cutoff_dist=None)
    base_seqs = np.concatenate([base_seqs, bighat_data[0]])
    return np.unique(base_seqs)


class LocalFLSampler(object):
    def optimize(self, top_k=1, *args, **kwargs):
        rand_seqs, rand_targets = draw_localfl_proposals(top_k)
        base_seqs = np.copy(rand_seqs)
        return rand_seqs, rand_targets, base_seqs


def draw_localfl_proposals(num_proposals):
    data, max_len, tag = ind_splits(
        "tape", "log_fluorescence", split=0.95, train_wo_cutoff=False
    )
    tape_seqs = np.concatenate([data[0], data[2]])
    tape_targets = np.concatenate([data[1], data[3]])
    brightness_cutoff = np.quantile(tape_targets, 0.75)
    tape_seqs = tape_seqs[tape_targets > brightness_cutoff]
    idxs = np.random.randint(0, tape_seqs.shape[0], (num_proposals,))
    return tape_seqs[idxs], tape_targets[idxs]


def make_dataset(dset_cfg):
    all_inputs, all_labels = [], []
    for source_dict in dset_cfg['sources']:
        print(f'[dataset] loading from {source_dict["name"]}')
        source_data = source_dict['_target_'](**source_dict['kwargs'])
        all_inputs.append(source_data[0])
        all_labels.append(source_data[1])

    all_inputs = np.concatenate(all_inputs)
    all_labels = np.concatenate(all_labels)
    num_total = all_inputs.shape[0]

    if dset_cfg['shuffle']:
        perm_idx = np.random.permutation(num_total)
        all_inputs = all_inputs[perm_idx]
        all_labels = all_labels[perm_idx]

    num_train = math.ceil(num_total * dset_cfg['split_ratio'])

    x_train, x_test = all_inputs[:num_train], all_inputs[num_train:]
    y_train, y_test = all_labels[:num_train], all_labels[num_train:]

    if dset_cfg['use_all']:
        print("WARNING the surrogate eval metrics will produced by 'testing on train'.")
        x_train = np.concatenate([x_train, x_test])
        y_train = np.concatenate([y_train, y_test])

    label_mean, label_std = y_train.mean(0), y_test.std(0)
    y_train = (y_train - label_mean) / label_std
    y_test = (y_test - label_mean) / label_std

    return (x_train, y_train), (x_test, y_test), (label_mean, label_std)


def count_params(module):
    num_params = 0
    for param in module.parameters():
        num_params += param.numel()
    return num_params


def generate_proposals(dataset_config, proposal_configs):
    _generate_proposals(dataset_config, proposal_configs, "Stokes Shift")


def main(**config):
    dataset_config = {
        'split_ratio': config.setdefault('split_ratio', 0.9),
        'shuffle': config.setdefault('shuffle', True),
        'use_all': config.setdefault("use_all", True),  # train surrogates on both train and test splits
        'sources': []
    }
    dataset_config['sources'].append({
        'name': 'fpbase',
        '_target_': utils._load_fpbase_data,
        'kwargs': dict(root='./bo_protein', task="Stokes Shift", cutoff_dist=20),
    })
    dataset_config['sources'].append({
        'name': 'bighat',
        '_target_': utils._load_bighat_data,
        'kwargs': dict(root='./bo_protein', task='ObservedStokesShift', cutoff_dist=None),
    })

    surrogate_config = {
            "model": config.setdefault("model", "CNN"),
            "model_kwargs": {'p': config.setdefault('dropout_prob', 0.)},
            "ensemble_size": config.setdefault("ensemble_size", 10),
            "num_epochs": config.setdefault("num_epochs", 400),
            "bootstrap_ratio": config.setdefault("bootstrap_ratio", None),
            "weight_decay": config.setdefault("weight_decay", 1e-4),
            "holdout_ratio": config.setdefault("holdout_ratio", 0.15),
            "early_stopping": config.setdefault("early_stopping", True),
            "lr": config.setdefault("lr", 1e-3),
            "bs": config.setdefault('bs', 32),
            'patience': config.setdefault('patience', 64),
            "max_shift": config.setdefault("max_shift", 4),
            "mask_size": config.setdefault("mask_size", 2),
        }
    acq_config = {
        'acq_type': config.setdefault('acq_type', 'ucb'),
        'ucb_beta': config.setdefault('ucb_beta', 0.2)
    }
    optimizer_kwargs = {
        'alphabet': bo_protein.utils.AMINO_ACIDS,
        'max_dist': config.setdefault("max_dist", 4),
        'num_evolutions': config.setdefault("num_evolutions", 30),
        'tournament_prob': config.setdefault("tournament_prob", 0.004),
        'population_size': config.setdefault("population_size", 1000),
        'p_crossover': config.setdefault("p_crossover", 1e-4),
        'p_mutation': config.setdefault("p_mutation", 5e-3)
    }

    proposal_configs = []
    # Embedding GP proposal config
    emb_gp_cfg = copy.deepcopy(surrogate_config)
    emb_gp_cfg.update({
        'finetune': config.setdefault('finetune', 'supervised'),
        'embedding_type': config.setdefault("embedding_type", "BERT")
    })
    emb_gp_tag = "_".join([emb_gp_cfg["embedding_type"].lower(), "emb_gp", acq_config["acq_type"]])
    emb_gp_tag = "_".join([emb_gp_tag, str(acq_config['ucb_beta'])]) if "ucb" in emb_gp_tag else emb_gp_tag
    proposal_configs.append({
        'tag': emb_gp_tag,
        'num_proposals': 40,
        'surrogate_constr': surrogates.EmbeddingGP,
        'surrogate_config': emb_gp_cfg,
        'acq_constr': acquisition.SingleFidelityAcquisition,
        'acq_config': acq_config,
        'optimizer_constr': LocalGeneticOptimizer,
        'optimizer_kwargs': optimizer_kwargs,
        'unique_base_seq': True,
    })
    # Ensemble proposal config
    ens_cfg = copy.deepcopy(surrogate_config)
    deep_ens_tag = "_".join([ens_cfg["model"].lower(), str(ens_cfg["ensemble_size"]), "ens", acq_config["acq_type"]])
    deep_ens_tag = "_".join([deep_ens_tag, str(acq_config['ucb_beta'])]) if "ucb" in deep_ens_tag else deep_ens_tag
    proposal_configs.append({
        'tag': deep_ens_tag,
        'num_proposals': 40,
        'surrogate_constr': surrogates.DeepEnsemble,
        'surrogate_config': ens_cfg,
        'acq_constr': acquisition.SingleFidelityAcquisition,
        'acq_config': acq_config,
        'optimizer_constr': LocalGeneticOptimizer,
        'optimizer_kwargs': optimizer_kwargs,
        'unique_base_seq': True,
    })
    # Control proposal config
    proposal_configs.append({
        'tag': 'control',
        'num_proposals': 15,
        'acq_constr': acquisition.ControlAcquisition,
        'optimizer_constr': LocalGeneticOptimizer,
        'optimizer_kwargs': optimizer_kwargs,
        'unique_base_seq': True
    })
    # LocalFL proposal config
    proposal_configs.append({
        'tag': 'localfl',
        'num_proposals': 0,
        'unique_base_seq': True,
        'optimizer_constr': LocalFLSampler,
        'optimizer_kwargs': {},
    })

    wandb.init(project="gfp-regression-nb", config=config)
    generate_proposals(dataset_config, proposal_configs)


if __name__ == "__main__":
    # os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

    from fire import Fire

    Fire(main)
