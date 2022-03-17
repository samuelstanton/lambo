import numpy as np
import selfies as sf
from pymoo.core.problem import Problem

from lambo.candidate import StringCandidate
from lambo.tasks.chem.logp import prop_func
from lambo.tasks.chem.utils import ChemWrapperModule, SELFIESTokenizer
from lambo.utils import apply_mutation, mutation_list


class ChemTask(Problem):
    # TODO this should be subclassing BaseTask
    def __init__(self, tokenizer, candidate_pool, obj_dim, obj_properties, num_start_examples=10000,
                 transform=lambda x: x, batch_size=1, candidate_weights=None, max_len=74, max_ngram_size=1,
                 worst_ratio=1., best_ratio=0., **kwargs):

        assert obj_dim == len(obj_properties), ''
        if len(candidate_pool) == 0:
            xl = 0.
            xu = 1.
        else:
            max_len = max([
                len(tokenizer.encode(cand.mutant_residue_seq)) - 2 for cand in candidate_pool
            ]) - 1
            self.op_types = ['sub', 'ins', 'del']
            if len(candidate_pool) == 0:
                xl = 0.
                xu = 1.
            else:
                xl = np.array([0] * 4 * batch_size)
                xu = np.array([
                    len(candidate_pool) - 1,  # base seq choice
                    2 * max_len,  # seq position choice
                    len(tokenizer.sampling_vocab) - 1,  # token choice
                    len(self.op_types) - 1,  # op choice
                ] * batch_size)

        n_var = 4 * batch_size
        super().__init__(
            n_var=n_var, n_obj=obj_dim, n_constr=0, xl=xl, xu=xu, type_var=int
        )
        self.tokenizer = tokenizer
        self.candidate_pool = list(candidate_pool)
        self.candidate_weights = candidate_weights
        self.transform = transform
        self.batch_size = batch_size
        self.num_start_examples = num_start_examples
        self.prop_func = prop_func
        self.obj_properties = obj_properties
        self.obj_dim = obj_dim
        self.op_types = ['sub', 'ins', 'del']
        self.max_len = max_len
        self.max_ngram_size = max_ngram_size
        self.allow_len_change = True
        self.worst_ratio = worst_ratio
        self.best_ratio = best_ratio

    def x_to_query_batches(self, x):
        return x.reshape(-1, self.batch_size, 4)

    def query_batches_to_x(self, query_batches):
        return query_batches.reshape(-1, self.n_var)

    def task_setup(self, *args, **kwargs):
        mod = ChemWrapperModule(self.num_start_examples, self.worst_ratio, self.best_ratio)
        all_seqs, all_targets = mod.sample_dataset(self.obj_properties)

        if isinstance(self.tokenizer, SELFIESTokenizer):
            all_seqs = np.array(
                list(map(sf.encoder, all_seqs))
            )

        base_candidates = np.array([
            StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer) for seq in all_seqs
        ]).reshape(-1)
        # all_targets = self.score(base_candidates)

        base_targets = all_targets.copy()
        return base_candidates, base_targets, all_seqs, all_targets

    def _evaluate(self, x, out, *args, **kwargs):
        assert x.ndim == 2
        x_cands, x_seqs, f_vals = [], [], []
        for query_pt in x:
            cand_idx, mut_pos, mut_res_idx, op_idx = query_pt
            op_type = self.op_types[op_idx]
            base_candidate = self.candidate_pool[cand_idx]
            base_seq = base_candidate.mutant_residue_seq
            mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
            mut_seq = apply_mutation(base_seq, mut_pos, mut_res, op_type, self.tokenizer)
            mutation_ops = mutation_list(base_seq, mut_seq, self.tokenizer)
            candidate = base_candidate.new_candidate(mutation_ops, self.tokenizer)
            x_cands.append(candidate)
            x_seqs.append(candidate.mutant_residue_seq)
        x_seqs = np.array(x_seqs).reshape(-1)
        x_cands = np.array(x_cands).reshape(-1)

        out["X_cand"] = x_cands
        out["X_seq"] = x_seqs
        out["F"] = self.transform(self.score(x_cands))

    def score(self, candidates):
        str_array = np.array([cand.mutant_residue_seq for cand in candidates])
        if isinstance(self.tokenizer, SELFIESTokenizer):
            smiles_strings = list(map(sf.decoder, str_array))
        else:
            smiles_strings = str_array

        scores = [self.prop_func(s, self.obj_properties) for s in smiles_strings]
        scores = -np.array(scores).astype(np.float64)
        return scores

    def is_feasible(self, candidates):
        scores = self.score(candidates)
        is_valid_mol = (scores[:, 0] < 100).reshape(-1)
        in_length = np.array([len(cand) <= self.max_len for cand in candidates]).reshape(-1)
        is_feasible = (is_valid_mol * in_length)
        return is_feasible

    def make_new_candidates(self, base_candidates, new_seqs):
        assert base_candidates.shape[0] == new_seqs.shape[0]
        new_candidates = []
        for b_cand, n_seq in zip(base_candidates, new_seqs):
            b_seq = b_cand.mutant_residue_seq
            mutation_ops = mutation_list(b_seq, n_seq, self.tokenizer)
            new_candidates.append(b_cand.new_candidate(mutation_ops, self.tokenizer))
        return np.stack(new_candidates)