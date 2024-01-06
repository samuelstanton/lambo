import numpy as np
import torch

from lambo.tasks.base_task import BaseTask
from lambo.utils import apply_mutation


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