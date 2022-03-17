import numpy as np

import torch

from pymoo.factory import get_mutation
from pymoo.core.mutation import Mutation

from lambo import utils
from lambo.tasks.chem.logp import prop_func
from lambo.models.mlm import sample_tokens


def get_mlm_mutation(mlm_obj, problem, cand_idx, res_idx):
    seqs = [problem.candidate_pool[i].mutant_residue_seq for i in cand_idx]
    base_tok_idxs = utils.str_to_tokens(seqs, mlm_obj.tokenizer)

    mask_idxs = res_idx.reshape(-1, 1)
    src_tok_idxs = base_tok_idxs.clone().to(mlm_obj.device)
    np.put_along_axis(src_tok_idxs, mask_idxs, mlm_obj.tokenizer.padding_idx, axis=1)

    with torch.no_grad():
        tgt_tok_logits, _ = mlm_obj.logits_from_tokens(src_tok_idxs)
    new_tok_idxs, _ = sample_tokens(
        base_tok_idxs, tgt_tok_logits, mlm_obj.tokenizer, replacement=False
    )
    new_tok_idxs = np.take_along_axis(new_tok_idxs, mask_idxs, axis=1).reshape(-1)
    new_toks = [mlm_obj.tokenizer.convert_id_to_token(t_idx) for t_idx in new_tok_idxs]
    sampling_vocab_idxs = np.array([
        mlm_obj.tokenizer.sampling_vocab.index(tok) for tok in new_toks
    ])
    return sampling_vocab_idxs


#following https://peerj.com/articles/pchem-11.pdf
def safe_vocab_mutation(tokenizer, problem, cand_idx, res_idx):
    muts = []
    seqs = [problem.candidate_pool[i].mutant_residue_seq for i in cand_idx]
    for seq, idx in zip(seqs, res_idx):
        tokens = tokenizer.decode(tokenizer.encode(seq)).split(" ")[1:-1]
        safe_mut = None
        for i in range(50):
            mut_idx = np.random.randint(0, len(tokenizer.sampling_vocab))
            mut_res = tokenizer.sampling_vocab[mut_idx]
            mut_seq = "".join(tokens[:idx] + [mut_res] + tokens[(idx + 1):])
            if prop_func(mut_seq) > -100:
                safe_mut = mut_idx
                break

        if safe_mut is None:
            muts.append(np.random.randint(0, len(tokenizer.sampling_vocab)))
        else:
            muts.append(safe_mut)

    return np.array(muts)


class UniformMutation(Mutation):
    def __init__(self, tokenizer=None, mlm_obj=None, safe_mut=False):
        self.tokenizer = tokenizer
        self.mlm_obj = mlm_obj
        self.safe_mut = safe_mut

    def _do(self, problem, x, **kwargs):
        query_batches = problem.x_to_query_batches(x)
        batch_shape, num_vars = query_batches.shape[:-1], query_batches.shape[-1]
        flat_queries = query_batches.reshape(-1, num_vars)
        num_samples = flat_queries.shape[0]

        x0 = flat_queries[..., 0]
        seqs = [problem.candidate_pool[i].mutant_residue_seq for i in x0]

        #NEXT LINE WON'T WORK UNLESS WE CHANGE CANDIDATE POOL TO NON-EMPTY IN TASK INIT
        x1 = np.random.randint(problem.xl[1], problem.xu[1], num_samples)
        x1 = np.array([idx % len(seq) for idx, seq in zip(x1, seqs)])

        if self.mlm_obj is None and not self.safe_mut:
            x2 = np.random.randint(0, len(self.tokenizer.sampling_vocab), num_samples)
        elif self.safe_mut:
            x2 = safe_vocab_mutation(self.tokenizer, problem, x0, x1)
        else:
            x2 = get_mlm_mutation(self.mlm_obj, problem, x0, x1)

        x3 = np.random.randint(0, len(problem.op_types), num_samples)

        new_queries = np.stack([x0, x1, x2, x3], axis=-1).reshape(*batch_shape, -1)
        new_x = problem.query_batches_to_x(new_queries)

        return new_x


class LocalMutation(Mutation):
    def __init__(self, eta, prob, tokenizer=None, mlm_obj=None, safe_mut=False):
        super().__init__()
        self.poly_mutation = get_mutation('int_pm', eta=eta, prob=prob)
        self.tokenizer = tokenizer
        self.mlm_obj = mlm_obj
        self.safe_mut = safe_mut

    def _do(self, problem, x, **kwargs):
        query_batches = problem.x_to_query_batches(x)
        batch_shape, num_vars = query_batches.shape[:-1], query_batches.shape[-1]
        flat_queries = query_batches.reshape(-1, num_vars)
        num_samples = flat_queries.shape[0]

        x0 = flat_queries[..., 0]
        seqs = [problem.candidate_pool[i].mutant_residue_seq for i in x0]

        mut_x = self.poly_mutation._do(problem, x)
        mut_x = problem.x_to_query_batches(mut_x).reshape(-1, num_vars)
        x1 = mut_x[..., 1]

        # x1 = np.array([idx % len(seq) for idx, seq in zip(x1, seqs)])

        for i, idx in enumerate(x0):
            num_tokens = len(self.tokenizer.encode(problem.candidate_pool[idx].mutant_residue_seq)) - 2
            x1[i] = min(num_tokens - 1, x1[i])
            # TODO always work with token indices?
            # num_tokens = len(self.tokenizer.encode(cand_seq))
            # x1[i] = min(num_tokens - 2, x1[i])  # skip end token
            # x1[i] = max(1, x1[i])  # skip start token

        if self.mlm_obj is None and not self.safe_mut:
            x2 = np.random.randint(0, len(self.tokenizer.sampling_vocab), num_samples)
        elif self.safe_mut:
            x2 = safe_vocab_mutation(self.tokenizer, problem, x0, x1)
        else:
            x2 = get_mlm_mutation(self.mlm_obj, problem, x0, x1)

        x3 = np.random.randint(0, len(problem.op_types), num_samples)

        new_queries = np.stack([x0, x1, x2, x3], axis=-1).reshape(*batch_shape, -1)
        new_x = problem.query_batches_to_x(new_queries)

        return new_x
