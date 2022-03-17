import numpy as np

from pymoo.core.crossover import Crossover


class BatchCrossover(Crossover):
    def __init__(self, prob, prob_per_query=0.5, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.prob_per_query = prob_per_query

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        offspring = np.copy(X)
        for mating_idx in range(n_matings):
            parents = X[:2, mating_idx]
            parent_batches = problem.x_to_query_batches(parents)
            batch_size = parent_batches.shape[-2]

            keep_mask = (np.random.rand(batch_size) > self.prob_per_query)

            child_1 = np.concatenate((parent_batches[:1, keep_mask], parent_batches[1:2, ~keep_mask]), axis=-2)
            child_2 = np.concatenate((parent_batches[:1, ~keep_mask], parent_batches[1:2, keep_mask]), axis=-2)
            child_batches = np.concatenate((child_1, child_2))
            child_x = problem.query_batches_to_x(child_batches)

            offspring[:2, mating_idx] = child_x

        if self.n_offsprings == 1:
            child_idx = np.random.randint(0, 2)
            offspring = offspring[child_idx].unsqueeze(0)

        return offspring
