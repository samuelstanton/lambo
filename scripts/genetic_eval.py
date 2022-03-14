import torch
import numpy as np
import wandb
import random

import bo_protein.utils
from bo_protein.utils import random_proteins
from scripts.regression import ind_splits
from bo_protein.rewards import RegexRewardFunc, EditDistRewardFunc
from bo_protein.optimizers.genetic import GeneticOptimizer, LocalGeneticOptimizer
from bo_protein.gfp_data import utils
from bo_protein.models import surrogates
from bo_protein import acquisition

import warnings
warnings.filterwarnings("ignore")


def evaluate_regex():
    start_pool = random_proteins(500)
    opt = GeneticOptimizer(
        bo_protein.utils.AMINO_ACIDS, max_len=500, num_evolutions=200, population_size=200
    )

    reward_func = RegexRewardFunc("(?=AA)")

    max_seq, max_reward = opt.optimize(
        start_pool, reward_func, log_prefix="genetic_opt/regex"
    )

    print("\n--Regex optimization--")
    print(f"Max sequence: {max_seq[0]}")
    print(f"Max score: {max_reward[0]}")


def nn_reward_func(config, source, task, start_from_random=False):
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    print("\n--NN reward optimization--")
    for i in range(5):
        np.random.seed(i)
        random.seed(i)

        data, max_len, tag = ind_splits(source, task)
        _config = {"ensemble_size": 5, "acq_type": "posterior-mean"}
        surrogate = surrogates.DeepEnsemble(_config)
        surrogate.fit(*data, log_prefix=tag)
        acq_fn = acquisition.SingleFidelityAcquisition(surrogate, _config)

        if start_from_random:
            start_pool = random_proteins(len(data[0]))
        else:
            start_pool = data[0]
        opt = LocalGeneticOptimizer(
            bo_protein.utils.AMINO_ACIDS,
            max_len=max_len - 2,  # subtract off start/end tokens
            max_dist=config.get("max_dist", 10),
            num_evolutions=config.get("num_evolutions", 200),
            tournament_prob=config.get("tournament_prob", 0.1),
            population_size=config.get("population_size", 1000),
            p_crossover=config.get("p_crossover", 0.01),
            p_mutation=config.get("p_mutation", 0.01),
        )

        max_seq, max_reward, _ = opt.optimize(
            start_pool, acq_fn, log_prefix=f"genetic_opt/{task}_surrogate/trial_{i}"
        )

        max_start_idx = np.argmax(acq_fn.score(start_pool))
        start_scores = acq_fn.score(start_pool)
        max_start_seq = start_pool[max_start_idx]
        max_start_score = start_scores[max_start_idx]

        print(f"Iteration {i}")
        print(f"\tMax starting sequence (truncated): {max_start_seq[:20]}")
        print(f"\tMax starting score: {max_start_score}\n")

        print(f"\tMax optimized sequence (truncated): {max_seq[0][:20]}")
        print(f"\tMax optimized score: {max_reward[0]}\n")


def edit_dist_reward_func(config):
    """Minimize Levenshtein distance from avgFP protein seq."""
    _, (start_pool, _, _) = utils.load_fpbase_data_mutation_split("Quantum Yield")
    max_len = max([len(x) for x in start_pool])
    opt = LocalGeneticOptimizer(
        bo_protein.utils.AMINO_ACIDS,
        max_len,
        max_dist=100,
        num_evolutions=config.get("num_evolutions", 200),
        tournament_prob=config.get("tournament_prob", 0.5),
        population_size=config.get("population_size", 200),
        p_crossover=config.get("p_crossover", 0.0),
        p_mutation=config.get("p_mutation", 0.01),
    )
    reward_func = EditDistRewardFunc(utils.avGFP)
    max_seq, max_reward, _ = opt.optimize(
        start_pool, reward_func, log_prefix="genetic_opt/avg_gfp_dist"
    )

    print("\n--Edit dist optimization--")
    print(f"Max sequence: {max_seq[0]}")
    print(f"Max score: {-max_reward[0]}")


def evaluate_gfp(config):
    # nn_reward_func(config, 'fpbase', 'Quantum Yield')
    nn_reward_func(config, "fpbase", "Stokes Shift")
    # nn_reward_func(config, 'tape', 'log_fluorescence')
    # edit_dist_reward_func(config)


def main(**cfg):
    wandb.init(project="gfp-regression-nb", config=cfg)
    evaluate_regex()
    evaluate_gfp(cfg)


if __name__ == "__main__":
    # os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

    from fire import Fire

    Fire(main)
