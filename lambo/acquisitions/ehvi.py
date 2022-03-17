import torch
from botorch.models import KroneckerMultiTaskGP
from botorch.sampling import IIDNormalSampler
from botorch.utils.multi_objective import infer_reference_point
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from lambo.acquisitions.monte_carlo import qDiscreteEHVI, qMTGPDiscreteNEHVI, qDiscreteNEHVI
from lambo.utils import batched_call


class EHVI(object):
    def __init__(self, surrogate, known_targets, num_samples,
                 batch_size, ref_point=None, **kwargs):
        self.ref_point = infer_reference_point(known_targets) if ref_point is None else ref_point
        sampler = IIDNormalSampler(num_samples=num_samples)
        partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=known_targets)
        acq_kwargs = dict(
            model=surrogate,
            ref_point=self.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        self.out_dim = 1
        self.batch_size = batch_size
        self.acq_fn = qDiscreteEHVI(**acq_kwargs)

    def __call__(self, candidates):
        acq_vals = torch.cat(
            batched_call(self.acq_fn, candidates, batch_size=1)
        )
        return acq_vals


class NoisyEHVI(EHVI):
    def __init__(self, surrogate, X_baseline, known_targets, num_samples,
                 batch_size, ref_point=None, **kwargs):
        self.ref_point = infer_reference_point(known_targets) if ref_point is None else ref_point
        sampler = IIDNormalSampler(num_samples=num_samples)
        acq_kwargs = dict(
            model=surrogate,
            ref_point=self.ref_point,
            sampler=sampler,
            X_baseline=X_baseline,
            prune_baseline=False,
        )
        self.out_dim = 1
        self.batch_size = batch_size
        if isinstance(surrogate, KroneckerMultiTaskGP):
            # TODO: remove when botorch #1037 goes in
            self.acq_fn = qMTGPDiscreteNEHVI(**acq_kwargs)
        else:
            self.acq_fn = qDiscreteNEHVI(**acq_kwargs)
