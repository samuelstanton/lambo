import numpy as np
import wandb
from Levenshtein import distance as edit_distance
import random
import pandas as pd

from tqdm import tqdm
from scipy.special import softmax as np_softmax

import bo_protein.utils
from bo_protein.gfp_data import utils


MUTATION_TYPES = ['insertion', 'deletion', 'substitution']
TYPE_WEIGHTS = [0.1, 0.1, 0.8]


def mutate(parent, mutation_prob):
    child = []
    for char in parent:
        if np.random.rand() < mutation_prob:
            m_type = np.random.choice(
                MUTATION_TYPES,
                p=TYPE_WEIGHTS,
            )
            if m_type == "deletion":
                continue
            elif m_type == "insertion":
                child.append(np.random.choice(bo_protein.utils.AMINO_ACIDS))
                child.append(char)
            elif m_type == "substitution":
                child.append(np.random.choice(bo_protein.utils.AMINO_ACIDS))
        else:
            child.append(char)
    return ''.join(child)


def crossover(parent_1, parent_2):
    min_len = min(len(parent_1), len(parent_2))
    cross_pt = np.random.randint(min_len)
    if np.random.rand() < 0.5:
        return parent_1[:cross_pt] + parent_2[cross_pt:]
    else:
        return parent_1[cross_pt:] + parent_2[:cross_pt]


def reproduce(parent_1, parent_2, crossover_prob, mutation_prob, max_len):
    if np.random.rand() < crossover_prob:
        parent_1 = crossover(parent_1, parent_2)
    child = mutate(parent_1, mutation_prob)
    return child[:max_len]


def draw_candidates(start_pool, num_samples, weights=None, replace=None):

    replace = True if start_pool.shape[0] < num_samples or replace else False
    rng = np.random.default_rng()
    pool_seqs = rng.choice(start_pool, size=num_samples, replace=replace, p=weights)

    pool_seqs = [x for x in pool_seqs]
    base_seqs, cand_seqs = [], []

    while len(cand_seqs) < num_samples:
        b_seq = random.choice(pool_seqs)
        m_type = random.choices(MUTATION_TYPES, weights=TYPE_WEIGHTS)[0]
        m_idx = random.randint(0, len(b_seq))
        char = random.choice(bo_protein.utils.AMINO_ACIDS)

        if m_type == 'substitution' and m_idx < len(b_seq) - 1:
            c_seq = b_seq[:m_idx] + char + b_seq[m_idx + 1:]
        elif m_type == 'substitution':
            c_seq = b_seq[:-1] + char
        elif m_type == 'deletion' and m_idx < len(b_seq) - 1:
            c_seq = b_seq[:m_idx] + b_seq[m_idx + 1:]
        elif m_type == 'deletion':
            c_seq = b_seq[:-1]
        elif m_type == 'insertion':
            c_seq = b_seq[:m_idx] + char + b_seq[m_idx:]
        else:
            raise RuntimeError('unrecognized mutation variant')

        if c_seq in cand_seqs:
            continue
        else:
            base_seqs.append(b_seq)
            cand_seqs.append(c_seq)

    return np.array(base_seqs), np.array(cand_seqs)


class GeneticOptimizer:
    def __init__(
        self,
        alphabet,
        max_len,
        early_stop=False,
        num_evolutions=10,
        population_size=10,
        tournament_prob=0.5,
        p_crossover=0.8,
        p_mutation=0.05,
        p_deletion=0.1,
        p_insertion=0.1,
        p_substitution=0.8,
        weight_pop=False,
    ):
        self.alphabet = alphabet
        self.max_len = max_len

        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.num_evolutions = num_evolutions
        self.population_size = population_size
        self.tournament_prob = tournament_prob
        self.reproduce_fn = np.vectorize(reproduce)
        norm_factor = sum([p_deletion, p_insertion, p_substitution])
        assert norm_factor > 0
        self.mutation_options = dict(
            deletion=p_deletion / norm_factor,
            insertion=p_insertion / norm_factor,
            substitution=p_substitution / norm_factor,
        )

        self.early_stop = early_stop
        self.weight_pop = weight_pop

    def optimize(
        self, start_pool, reward_func, top_k=1, pool_scores=None, per_iter_log=False, log_prefix="", it=0
    ):
        self.start_pool = start_pool
        weights = None if pool_scores is None else np_softmax(pool_scores.reshape(-1))
        weights = weights if self.weight_pop else None
        _, population = draw_candidates(start_pool, self.population_size, weights)
        if hasattr(reward_func, 'set_candidates'):
            _, ref_candidates = draw_candidates(start_pool, self.population_size, weights)
            reward_func.set_candidates(ref_candidates)
        fitness_pop = reward_func.score(population)

        max_idxs = np.argsort(-fitness_pop)[:top_k]
        X_max = population[max_idxs]
        fitness_max = fitness_pop[max_idxs]

        if per_iter_log:
            scores = []
            proposals = []

        records = [dict(seq=x, label=y, batch=0, it=it) for x, y in zip(population, fitness_pop)]

        for step in range(self.num_evolutions):
            population = self._evolve(population, fitness_pop)
            # pop_weights = np_softmax(fitness_pop)
            # _, population = draw_candidates(population, self.population_size, pop_weights, replace=True)

            fitness_pop = reward_func.score(population)

            records.extend([dict(seq=x, label=y, batch=step + 1, it=it) for x, y in zip(population, fitness_pop)])

            combined_pop = np.concatenate([population, X_max])
            combined_fitness = np.concatenate([fitness_pop, fitness_max])

            _, unique_idx = np.unique(combined_pop, return_index=True)

            # print(len(combined_fitness))
            # print(unique_idx)

            combined_pop = combined_pop[unique_idx]
            combined_fitness = combined_fitness[unique_idx]

            max_idxs = np.argsort(-combined_fitness)[:top_k]
            _X_max = combined_pop[max_idxs]
            fitness_max = combined_fitness[max_idxs]

            # print(_X_max)
            # print(fitness_max)

            if per_iter_log:
                scores.append(fitness_max)
                proposals.append(_X_max)

            if self.early_stop and step > 10 and np.array_equal(X_max, _X_max):
                break

            X_max = _X_max

            if isinstance(log_prefix, str):
                metrics = {
                    "/".join([log_prefix, "pop_score_mean"]): np.mean(fitness_pop),
                    "/".join([log_prefix, "pop_score_max"]): np.max(fitness_pop),
                    "/".join([log_prefix, "pop_score_min"]): np.min(fitness_pop),
                    # "/".join([log_prefix, "pop_size"]): population.shape[0],
                    "/".join([log_prefix, "pop_perc_unique"]): float(
                        len(set(population))
                    )
                                                               / len(population),
                    "/".join([log_prefix, "best_score"]): fitness_max[0],
                }
                try:
                    wandb.log(metrics)
                except Exception as e:
                    print(e)

        idx = np.argsort(-fitness_max)
        X_max = X_max[idx]
        fitness_max = fitness_max[idx]

        history = pd.DataFrame(records)
        # history.to_csv(f'notebooks/gp_shift_analysis/genetic_history_{it}.csv', index=False)

        if per_iter_log:
            return X_max, fitness_max, (scores, proposals)

        return X_max, fitness_max

    def _evolve(self, population, fitness):
        gold_idxs, silver_idxs = self._tournament(fitness)
        gold_parents = population[gold_idxs]
        silver_parents = population[silver_idxs]
        new_pop = self.reproduce_fn(gold_parents, silver_parents, self.p_crossover, self.p_mutation,
                                      self.max_len)
        return new_pop

    def _tournament(self, fitness):
        tourn_size = int(self.population_size * self.tournament_prob)
        tourn_size = max(tourn_size, 2)
        contender_indices = np.random.randint(self.population_size, size=(self.population_size, tourn_size))
        contender_fitness = fitness[contender_indices]
        best = np.argmax(contender_fitness, axis=-1)
        contender_fitness[np.arange(self.population_size), best] = np.min(contender_fitness, axis=-1)
        second_best = np.argmax(contender_fitness, axis=-1)
        gold_idxs = contender_indices[np.arange(self.population_size), best]
        silver_idxs = contender_indices[np.arange(self.population_size), second_best]
        return gold_idxs, silver_idxs


class LocalGeneticOptimizer(GeneticOptimizer):
    def __init__(
        self,
        alphabet,
        max_len,
        max_dist,
        early_stop=False,
        num_evolutions=10,
        population_size=10,
        tournament_prob=0.5,
        p_crossover=0.8,
        p_mutation=0.05,
        p_deletion=0.1,
        p_insertion=0.1,
        p_substitution=0.8,
        weight_pop=False
    ):
        self.alphabet = alphabet
        self.max_len = max_len
        self.max_dist = max_dist

        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.num_evolutions = num_evolutions
        self.population_size = population_size
        self.tournament_prob = tournament_prob
        norm_factor = sum([p_deletion, p_insertion, p_substitution])
        assert norm_factor > 0
        self.mutation_options = dict(
            deletion=p_deletion / norm_factor,
            insertion=p_insertion / norm_factor,
            substitution=p_substitution / norm_factor,
        )

        self.early_stop = early_stop
        self.weight_pop = weight_pop

    def optimize(
        self, start_pool, reward_func, top_k=1, pool_scores=None, per_iter_log=False, log_prefix=""
    ):
        self.start_pool = start_pool
        weights = None if pool_scores is None else np_softmax(pool_scores.reshape(-1))
        weights = weights if self.weight_pop else None
        ancestors, population = draw_candidates(start_pool, self.population_size, weights)
        if hasattr(reward_func, 'set_candidates'):
            _, ref_candidates = draw_candidates(start_pool, self.population_size, weights)
            reward_func.set_candidates(ref_candidates)
        fitness_pop = reward_func.score(population)

        max_idxs = np.argsort(-fitness_pop)[:top_k]
        ancestor_max = ancestors[max_idxs]
        X_max = population[max_idxs]
        fitness_max = fitness_pop[max_idxs]

        if per_iter_log:
            scores = []
            proposals = []

        for step in range(self.num_evolutions):
            population, ancestors = self._evolve(population, fitness_pop, ancestors)
            avg_dist = np.stack(
                [edit_distance(x, y) for x, y in zip(population, ancestors)]
            ).mean()

            fitness_pop = reward_func.score(population)

            combined_pop = np.concatenate([population, X_max])
            combined_fitness = np.concatenate([fitness_pop, fitness_max])
            combined_ancestors = np.concatenate([ancestors, ancestor_max])

            _, unique_idx = np.unique(combined_pop, return_index=True)

            combined_pop = combined_pop[unique_idx]
            combined_fitness = combined_fitness[unique_idx]
            combined_ancestors = combined_ancestors[unique_idx]

            max_idxs = np.argsort(-combined_fitness)[:top_k]
            ancestor_max = combined_ancestors[max_idxs]
            _X_max = combined_pop[max_idxs]
            fitness_max = combined_fitness[max_idxs]

            # reintroduce start pool samples if optimization stagnates
            if avg_dist >= self.max_dist - 1:
                refresh_ancestors, refresh_pop = draw_candidates(start_pool, self.population_size, weights)
                refresh_score = reward_func.score(refresh_pop)
                population = np.concatenate([refresh_pop, population])
                fitness_pop = np.concatenate([refresh_score, fitness_pop])
                ancestors = np.concatenate([refresh_ancestors, ancestors])

            if per_iter_log:
                scores.append(fitness_max)
                proposals.append(_X_max)

            if self.early_stop and step > 10 and np.array_equal(X_max, _X_max):
                break

            X_max = _X_max

            if isinstance(log_prefix, str):
                metrics = {
                    # "/".join([log_prefix, "pop_score_mean"]): np.mean(fitness_pop),
                    # "/".join([log_prefix, "pop_score_max"]): np.max(fitness_pop),
                    # "/".join([log_prefix, "pop_score_min"]): np.min(fitness_pop),
                    # "/".join([log_prefix, "pop_size"]): population.shape[0],
                    "/".join([log_prefix, "pop_avg_dist"]): avg_dist,
                    "/".join([log_prefix, "pop_perc_unique"]): float(
                        len(set(population))
                    )
                    / len(population),
                    "/".join([log_prefix, "best_score"]): fitness_max[0],
                }
                try:
                    wandb.log(metrics)
                except Exception as e:
                    print(e)

        idx = np.argsort(-fitness_max)
        X_max = X_max[idx]
        fitness_max = fitness_max[idx]
        ancestor_max = ancestor_max[idx]

        if per_iter_log:
            return X_max, fitness_max, ancestor_max, (scores, proposals)

        return X_max, fitness_max, ancestor_max

    def _crossover_then_mutate(self, parent1, parent2):
        parent1 = list(parent1)
        parent2 = list(parent2)

        min_len = min(len(parent1), len(parent2))
        cross_pt = np.random.randint(min_len)

        temp = parent1[:cross_pt]
        parent1[:cross_pt] = parent2[:cross_pt]
        parent2[:cross_pt] = temp

        for i in range(min_len):
            if i < len(parent1) and np.random.rand() < self.p_mutation:
                parent1[i] = np.random.choice(self.alphabet)

            if i < len(parent2) and np.random.rand() < self.p_mutation:
                parent2[i] = np.random.choice(self.alphabet)
        child1 = "".join(parent1[: self.max_len])
        child2 = "".join(parent2[: self.max_len])
        return child1, child2

    def _reproduce_then_mutate(self, parent):
        parent = list(parent)
        child = []
        for i in range(len(parent)):
            if np.random.rand() < self.p_mutation:
                m_type = np.random.choice(
                    list(self.mutation_options.keys()),
                    p=list(self.mutation_options.values()),
                )
                if m_type == "deletion":
                    continue
                elif m_type == "insertion":
                    child.append(np.random.choice(self.alphabet))
                    child.append(parent[i])
                elif m_type == "substitution":
                    child.append(np.random.choice(self.alphabet))
            else:
                child.append(parent[i])

        return "".join(child[: self.max_len])

    def _evolve(self, population, fitness, ancestors):
        new_pop = []
        new_ancestors = []

        i = 0
        while i < self.population_size:
            tournament = self._tournament(fitness)

            parent1 = population[tournament[0][0]]
            parent2 = population[tournament[1][0]]
            ancestor1 = ancestors[tournament[0][0]]
            ancestor2 = ancestors[tournament[1][0]]

            if np.random.rand() < self.p_crossover:
                child1, child2 = self._crossover_then_mutate(parent1, parent2)
                dist_1 = np.array(
                    [
                        edit_distance(ancestor1, child1),
                        edit_distance(ancestor2, child1),
                    ]
                )
                dist_2 = np.array(
                    [
                        edit_distance(ancestor1, child2),
                        edit_distance(ancestor2, child2),
                    ]
                )
                if dist_1.min() <= self.max_dist:
                    new_pop.append(child1)
                    new_ancestors.append(ancestors[tournament[dist_1.argmin()][0]])
                    i += 1
                if i < len(population) - 1 and dist_2.min() <= self.max_dist:
                    new_pop.append(child2)
                    new_ancestors.append(ancestors[tournament[dist_2.argmin()][0]])
                    i += 1
            else:
                child1 = self._reproduce_then_mutate(parent1)
                if edit_distance(ancestor1, child1) <= self.max_dist:
                    new_pop.append(child1)
                    new_ancestors.append(ancestor1)
                    i += 1

        new_pop = np.array(new_pop)
        new_ancestors = np.array(new_ancestors)

        return new_pop, new_ancestors
