import re
import hydra
import numpy as np
import pandas as pd
import torch
import selfies as sf

from pymoo.core.problem import Problem

from botorch.utils.multi_objective import pareto

import bo_protein.utils
from bo_protein.gfp_data import utils
from bo_protein.candidate import (
    FoldedCandidate,
    StringCandidate,
#     SMILESCandidate
)
from bo_protein.utils import random_proteins, mutation_list
from bo_protein.chem_data.utils import ChemWrapperModule, prop_func, SELFIESTokenizer


def apply_mutation(base_seq, mut_pos, mut_res, op_type, tokenizer):
    tokens = tokenizer.decode(tokenizer.encode(base_seq)).split(" ")[1:-1]

    if op_type == 'sub':
        mut_seq = "".join(tokens[:mut_pos] + [mut_res] + tokens[(mut_pos + 1):])
    elif op_type == 'ins':
        mut_seq = "".join(tokens[:mut_pos] + [mut_res] + tokens[mut_pos:])
    elif op_type == 'del':
        mut_seq = "".join(tokens[:mut_pos] + tokens[(mut_pos + 1):])
    else:
        raise ValueError('unsupported operation')

    return mut_seq


class BaseTask(Problem):
    def __init__(self, tokenizer, candidate_pool, obj_dim, transform=lambda x: x, batch_size=1,
                 candidate_weights=None, max_len=None, max_ngram_size=1, allow_len_change=True, **kwargs):
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
        self.obj_dim = obj_dim
        self.transform = transform
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_ngram_size = max_ngram_size
        self.allow_len_change = allow_len_change

    def make_new_candidates(self, base_candidates, new_seqs):
        assert base_candidates.shape[0] == new_seqs.shape[0]
        new_candidates = []
        for b_cand, n_seq in zip(base_candidates, new_seqs):
            mutation_ops = mutation_list(
                b_cand.mutant_residue_seq,
                n_seq,
                self.tokenizer
            )
            new_candidates.append(b_cand.new_candidate(mutation_ops, self.tokenizer))
        return np.stack(new_candidates)

    def task_setup(self, *args, **kwargs):
        raise NotImplementedError

    def x_to_query_batches(self, x):
        return x.reshape(-1, self.batch_size, 4)

    def query_batches_to_x(self, query_batches):
        return query_batches.reshape(-1, self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        raise NotImplementedError

    def score(self, str_array):
        raise NotImplementedError

    def is_feasible(self, candidates):
        if self.max_len is None:
            is_feasible = np.ones(candidates.shape).astype(bool)
        else:
            is_feasible = np.array([len(cand) <= self.max_len for cand in candidates]).reshape(-1)
        return is_feasible


class ProxyRFPTask(BaseTask):
    def __init__(self, max_len, tokenizer, candidate_pool, obj_dim, transform=lambda x: x,
                 num_start_examples=1024, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim, transform, **kwargs)
        self.max_len = max_len
        self.op_types = ["sub"]
        self.num_start_examples = num_start_examples

    def task_setup(self, config, project_root=None, *args, **kwargs):
        project_root = hydra.utils.get_original_cwd() if project_root is None else project_root
        work_dir = f'{project_root}/{config.log_dir}/{config.job_name}/{config.timestamp}/foldx'
        rfp_known_structures = pd.read_csv(
            f'{project_root}/bo_protein/assets/fpbase/rfp_known_structures.csv'
        )

        all_seqs = rfp_known_structures.foldx_seq.values
        all_targets = np.stack([
            -rfp_known_structures.SASA.values,
            rfp_known_structures.foldx_total_energy.values,
        ], axis=-1)

        seed_data = pd.read_csv(
            f'{project_root}/bo_protein/assets/fpbase/proxy_rfp_seed_data.csv'
        )
        seed_data = seed_data.sample(self.num_start_examples)
        sample_batch_targets = np.stack([
            -seed_data.SASA.values,
            -seed_data.stability.values,
        ], axis=-1)

        all_seqs = np.concatenate((all_seqs, seed_data.foldx_seq.values))
        all_targets = np.concatenate((all_targets, sample_batch_targets))

        seq_len_mask = np.array([len(x) <= self.max_len for x in all_seqs])
        all_seqs = all_seqs[seq_len_mask]
        all_targets = all_targets[seq_len_mask]

        # filter candidate sequences by length
        foldx_seq_len = rfp_known_structures.foldx_seq.apply(lambda x: len(x))
        rfp_known_structures = rfp_known_structures[foldx_seq_len <= self.max_len]
        rfp_known_structures.reset_index(inplace=True)

        # find valid, non-dominated starting candidates
        # valid_seqs = rfp_known_structures.foldx_seq.values
        valid_targets = np.stack([
            -rfp_known_structures.SASA.values,
            rfp_known_structures.foldx_total_energy.values,
        ], axis=-1)
        pareto_mask = pareto.is_non_dominated(-torch.tensor(valid_targets))
        base_targets = valid_targets[pareto_mask]

        base_candidates = []
        for row_idx, datum in rfp_known_structures.iterrows():
            if not pareto_mask[row_idx]:
                continue
            print(f'{datum.Name} is non-dominated, adding to start pool')
            pdb_id = datum.pdb_id.lower()
            chain_id = datum.longest_chain
            parent_pdb_path = f'{project_root}/bo_protein/assets/foldx/{pdb_id}_{chain_id}/wt_input_Repair.pdb'
            base_candidates.append(
                FoldedCandidate(work_dir, parent_pdb_path, [], self.tokenizer,
                                skip_minimization=True, chain=chain_id, wild_name=datum.Name)
            )
        base_candidates = np.array(base_candidates).reshape(-1)

        return base_candidates, base_targets, all_seqs, all_targets

    def make_new_candidates(self, base_candidates, new_seqs):
        assert base_candidates.shape[0] == new_seqs.shape[0]
        new_candidates = []
        for b_cand, n_seq in zip(base_candidates, new_seqs):
            b_seq = b_cand.mutant_residue_seq
            assert len(b_seq) == len(n_seq), 'FoldX only accepts substitutions'
            mutation_ops = []
            for i, (b_char, n_char) in enumerate(zip(b_seq, n_seq)):
                if not b_char == n_char:
                    mutation_ops.append(b_cand.new_mutation(i, n_char, 'sub'))
            # mutation_ops = mutation_list(b_cand.mutant_residue_seq, n_seq)
            new_candidates.append(b_cand.new_candidate(mutation_ops))
        return np.stack(new_candidates)

    def _evaluate(self, x, out, *args, **kwargs):
        assert x.ndim == 2
        x_cands, x_seqs, f_vals = [], [], []
        for query_pt in x:
            cand_idx, mut_pos, mut_res_idx, _ = query_pt
            base_candidate = self.candidate_pool[cand_idx]
            mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
            mut_list = [base_candidate.new_mutation(mut_pos, mut_res, mutation_type='sub')]
            candidate = base_candidate.new_candidate(mut_list)
            x_cands.append(candidate)
            x_seqs.append(candidate.mutant_residue_seq)
        x_seqs = np.array(x_seqs).reshape(-1)
        x_cands = np.array(x_cands).reshape(-1)

        out["X_cand"] = x_cands
        out["X_seq"] = x_seqs
        norm_scores = self.transform(self.score(x_cands))
        out["F"] = norm_scores

    def score(self, candidates):
        f_vals = []
        for cand in candidates:
            f_vals.append(np.array([
                -cand.mutant_surface_area,
                cand.mutant_total_energy,
            ]))
        return np.stack(f_vals)


class SurrogateTask(BaseTask):
    def __init__(self, tokenizer, candidate_pool, acq_fn, transform=lambda x: x, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim=acq_fn.out_dim,
                         transform=transform, **kwargs)
        self.acq_fn = acq_fn

    def _evaluate(self, x, out, *args, **kwargs):
        query_batches = self.x_to_query_batches(x)
        batch_shape, num_vars = query_batches.shape[:-1], query_batches.shape[-1]
        candidates = []
        for query_pt in query_batches.reshape(-1, num_vars):
            cand_idx, mut_pos, mut_res_idx, op_idx = query_pt
            op_type = self.op_types[op_idx]
            base_seq = self.candidate_pool[cand_idx].mutant_residue_seq
            mut_pos = mut_pos % len(base_seq)
            mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
            mutant_seq = apply_mutation(base_seq, mut_pos, mut_res, op_type, self.tokenizer)
            candidates.append(mutant_seq)
        candidates = np.array(candidates).reshape(*batch_shape)  # (pop_size, batch_size)
        with torch.inference_mode():
            acq_vals = self.acq_fn(candidates).cpu().numpy()
        out["F"] = -acq_vals


class RegexTask(BaseTask):
    def __init__(self, regex_list, min_len, max_len, num_start_examples, tokenizer,
                 candidate_pool, obj_dim, transform=lambda x: x, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim, transform, **kwargs)
        self.regex_list = regex_list
        self.min_len = min_len
        self.max_len = max_len
        self.num_start_examples = num_start_examples

    def task_setup(self, *args, **kwargs):
        num_examples = 0
        selected_seqs = []
        selected_targets = []
        while num_examples < self.num_start_examples:
            # account for start and stop tokens
            all_seqs = random_proteins(self.num_start_examples, self.min_len, self.max_len - 2)
            base_candidates = np.array([
                StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer) for seq in all_seqs
            ]).reshape(-1)
            all_targets = self.score(base_candidates)
            positive_example_mask = (all_targets < 0).sum(-1).astype(bool)
            num_positive = positive_example_mask.astype(int).sum()
            num_negative = all_targets.shape[0] - num_positive
            num_selected = min(num_positive, num_negative)

            selected_seqs.append(all_seqs[positive_example_mask][:num_selected])
            selected_targets.append(all_targets[positive_example_mask][:num_selected])
            selected_seqs.append(all_seqs[~positive_example_mask][:num_selected])
            selected_targets.append(all_targets[~positive_example_mask][:num_selected])
            num_examples += num_selected

        all_seqs = np.concatenate(selected_seqs)[:self.num_start_examples]
        all_targets = np.concatenate(selected_targets)[:self.num_start_examples]

        base_candidates = np.array([
            StringCandidate(seq, mutation_list=[], tokenizer=self.tokenizer) for seq in all_seqs
        ]).reshape(-1)
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
            # TODO add support for insertion and deletion here
            # mutation_ops = [base_candidate.new_mutation(mut_pos, mut_res, mutation_type='sub')]
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
        scores = []
        for regex in self.regex_list:
            scores.append(np.array([
                len(re.findall(regex, str(x))) for x in str_array
            ]).reshape(-1))
        scores = -np.stack(scores, axis=-1).astype(np.float64)
        return scores


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
