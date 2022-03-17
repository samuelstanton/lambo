from botorch.acquisition import qExpectedImprovement
from botorch.sampling import IIDNormalSampler


class BatchEI(object):
    def __init__(self, surrogate, known_targets, ref_point, num_samples,
                 batch_size, **kwargs):
        self.ref_point = ref_point
        sampler = IIDNormalSampler(num_samples=num_samples)
        self.acq_fn = qExpectedImprovement(
            model=surrogate,
            best_f=0.9 * known_targets.max(),
            sampler=sampler,
        )
        self.out_dim = 1
        self.batch_size = batch_size

    def __call__(self, candidates):
        acq_vals = self.acq_fn(candidates)
        return acq_vals
