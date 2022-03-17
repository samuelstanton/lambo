import numpy as np
from pymoo.core.sampling import Sampling


from lambo.optimizers.mutation import get_mlm_mutation, safe_vocab_mutation


def _draw_samples(tokenizer, cand_pool, problem, num_samples, mlm_obj=None, safe_mut=False):
    cand_weights = problem.candidate_weights
    if cand_weights is None:
        x0 = np.random.choice(
            np.arange(len(cand_pool)), num_samples, replace=True
        )
    else:
        x0 = np.random.choice(
            np.arange(len(cand_pool)), num_samples, p=cand_weights, replace=True
        )
    # don't sample start or end token indexes
    x1 = []
    for idx in x0:
        num_tokens = len(tokenizer.encode(cand_pool[idx].mutant_residue_seq)) - 2
        x1.append(np.random.randint(0, num_tokens))
        # TODO always work with token indices?
        # num_tokens = len(tokenizer.encode(cand_pool[idx].mutant_residue_seq))
        # x1.append(np.random.randint(1, num_tokens - 1))
    x1 = np.array(x1)

    if mlm_obj is None and not safe_mut:
        x2 = np.random.randint(0, len(tokenizer.sampling_vocab), num_samples)
    elif safe_mut:
        x2 = safe_vocab_mutation(tokenizer, problem, x0, x1)
    else:
        x2 = get_mlm_mutation(mlm_obj, problem, x0, x1)

    x3 = np.random.randint(0, len(problem.op_types), num_samples)

    return np.stack([x0, x1, x2, x3], axis=-1)


class CandidateSampler(Sampling):
    def __init__(self, tokenizer=None, var_type=np.float64, mlm_obj=None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.var_type = var_type
        self.mlm_obj = mlm_obj

    def _do(self, problem, n_samples, *args, **kwargs):
        cand_pool = problem.candidate_pool
        x = _draw_samples(self.tokenizer, cand_pool, problem, n_samples, self.mlm_obj)
        return x


class BatchSampler(CandidateSampler):
    def __init__(self, batch_size, tokenizer=None, var_type=np.float64, mlm_obj=None) -> None:
        super().__init__(var_type)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.var_type = var_type
        self.mlm_obj = mlm_obj

    def _do(self, problem, n_samples, *args, **kwargs):
        cand_pool = problem.candidate_pool
        batches = np.stack([
            _draw_samples(self.tokenizer, cand_pool, problem, self.batch_size, self.mlm_obj) for _ in range(n_samples)
        ])
        x = problem.query_batches_to_x(batches)
        return x
