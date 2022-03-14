import sys

sys.path.append("..")
sys.path.append(".")
import numpy as np
from bo_protein.optimizers.genetic import GeneticOptimizer, LocalGeneticOptimizer


class Count:
    def __init__(self, pattern):
        self.pattern = pattern

    def score(self, X):
        scores = [x.count(self.pattern) + 1 for x in X]
        return np.array(scores)


def test_genetic_optimizer():
    start_pool = np.array(["bbb", "bba"])
    alphabet = ["a", "b"]
    reward = Count("a")

    opt = GeneticOptimizer(alphabet, max_len=3)
    best, _ = opt.optimize(start_pool, reward, log_prefix=None)
    assert best == "aaa"

    opt = GeneticOptimizer(
        alphabet, max_len=4, num_evolutions=100, p_insertion=0.5, p_substitution=0.5
    )
    best, _ = opt.optimize(start_pool, reward, log_prefix=None)
    assert best == "aaaa"


def test_local_optimizer():
    start_pool = np.array(["bbb", "abbab"])
    alphabet = ["a", "b"]
    reward = Count("aba")
    opt = LocalGeneticOptimizer(
        alphabet,
        max_len=6,
        max_dist=2,
        population_size=100,
        num_evolutions=100,
        p_mutation=0.25,
        p_crossover=0.1,
        p_insertion=0.5,
        p_substitution=0.5,
    )
    best, score, _ = opt.optimize(start_pool, reward, log_prefix=None)
    assert best == "abaaba"

    start_pool = np.array(["b", "a"])
    reward = Count("a")
    opt = LocalGeneticOptimizer(
        alphabet,
        max_len=6,
        max_dist=2,
        population_size=100,
        num_evolutions=100,
        p_mutation=0.25,
        p_crossover=0.1,
        p_insertion=0.5,
        p_substitution=0.5,
    )
    best, score, _ = opt.optimize(start_pool, reward, log_prefix=None)
    assert best == "aaa"
