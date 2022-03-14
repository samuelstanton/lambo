import torch
import numpy as np

from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition import qExpectedImprovement
from botorch.utils.multi_objective import infer_reference_point, is_non_dominated
from botorch.models import KroneckerMultiTaskGP

from bo_protein.acquisitions.monte_carlo import qDiscreteEHVI, qDiscreteNEHVI, qMTGPDiscreteNEHVI
from bo_protein.utils import batched_call


class ExpectedObjVal(object):
    def __init__(self, surrogate, num_samples, obj_dim, **kwargs):
        self.surrogate = surrogate
        self.num_samples = num_samples
        self.out_dim = obj_dim
        self.batch_size = 1

    def __call__(self, candidates):
        if torch.is_tensor(candidates):
            pass
        # elif candidates.ndim == 1:
        #     pass
        if candidates.shape[-1] == 1:
            candidates = candidates.squeeze(-1)
        # else:
        #     raise RuntimeError('unexpected candidate array shape')

        pred_samples = self.surrogate.posterior(candidates).rsample(torch.Size([self.num_samples]))
        # pred_samples, _, _ = self.surrogate.predict(
        #     candidates, num_samples=self.num_samples, latent=True
        # )
        return pred_samples.mean(0)


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
        # print(f'\nacq. pareto front:\n{known_targets}')
        # print(f'acq. ref point:\n{self.ref_point}\n')
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


class BatchEI(object):
    def __init__(self, surrogate, known_targets, ref_point, num_samples,
                 batch_size, **kwargs):
        self.ref_point = ref_point
        sampler = IIDNormalSampler(num_samples=num_samples)
        self.acq_fn = qExpectedImprovement(
            model=surrogate,
            best_f=0.9 * known_targets,
            # best_f=known_targets.max(),
            sampler=sampler,
        )
        self.out_dim = 1
        self.batch_size = batch_size

    def __call__(self, candidates):
        acq_vals = self.acq_fn(candidates)
        return acq_vals


class ControlAcquisition(object):
    def score(self, X, **kwargs):
        return np.random.rand(X.shape[0])


class SingleFidelityAcquisition(torch.nn.Module):
    def __init__(self, surrogate, config):
        super().__init__()
        self.surrogate = surrogate
        self.config = config
        self.acq_type = config.get("acq_type", "posterior-mean")
        self.ucb_beta = config.get("ucb_beta", 0.1)

    def forward(self, *args, **kwargs):
        return self.surrogate(*args, **kwargs)

    def score(self, X, bs=100, **kwargs):
        with torch.no_grad():
            surrogate_out = self(X, bs=bs, **kwargs)
        samples, mean, std = [tens.cpu().numpy() if not isinstance(tens, np.ndarray) else tens
                                for tens in surrogate_out]

        if self.acq_type == "posterior-mean":
            return mean
        elif self.acq_type == "ucb":
            return mean + self.ucb_beta * std
        else:
            return samples


class DiscreteFidelityAcquisition(torch.nn.Module):
    def __init__(self, surrogate, cost_fn, noise_model, config):
        super().__init__()
        self.surrogate = surrogate
        self.cost_fn = cost_fn
        self.noise_model = noise_model
        self.query_fidelity_fn = np.vectorize(lambda q_point, obs_counts: obs_counts.get(q_point, 0))
        self.candidates = None
        self.cand_score_samples = None
        self.obs_counts = {}
        self.config = config

    def forward(self, *args, **kwargs):
        return self.surrogate(*args, **kwargs)

    def set_candidates(self, candidates):
        self.candidates = candidates
        self.cand_score_samples, _, _ = self(self.candidates, bs=256)

    def observe(self, X):
        for obs in X:
            self.obs_counts.setdefault(obs, 0)
            self.obs_counts[obs] += 1

    def score(self, query_set, bs=256, **kwargs):
        query_score_samples, _, _ = self(query_set, bs=bs)
        num_samples = self.config.get('num_noise_samples', 32)

        cand_fidelity = self.query_fidelity_fn(self.candidates, self.obs_counts)
        cand_noise = self.noise_model(self.candidates) / np.sqrt(cand_fidelity + 1) * \
                     np.random.normal(size=(num_samples, *self.cand_score_samples.shape))
        cand_f_samples = self.cand_score_samples + cand_noise

        noise_samples = np.random.normal(size=(num_samples, *query_score_samples.shape))
        pre_query_fidelity = self.query_fidelity_fn(query_set, self.obs_counts)
        pre_query_entropy = self.get_entropy(query_set, pre_query_fidelity, query_score_samples,
                                             cand_f_samples, noise_samples)

        post_query_fidelity = pre_query_fidelity + 1  # for now assume we can only query one-at-a-time
        post_query_entropy = self.get_entropy(query_set, post_query_fidelity, query_score_samples,
                                              cand_f_samples, noise_samples)

        query_cost = self.cost_fn(post_query_fidelity - pre_query_fidelity)
        acq_value = (pre_query_entropy - post_query_entropy) / query_cost
        return acq_value

    def get_entropy(self, query_set, query_fidelity, query_score_samples, cand_f_samples, noise_samples):
        query_noise = self.noise_model(query_set) / np.sqrt(query_fidelity + 1)
        query_f_samples = query_score_samples + np.sqrt(query_noise) * noise_samples

        cand_f_samples = np.tile(cand_f_samples, (query_f_samples.shape[-1], 1, 1, 1)).transpose(-1, 1, 2, 0)

        q_is_max_prob = (query_f_samples > cand_f_samples).astype(float).prod(0).mean(0).mean(0)
        prob_mask = (q_is_max_prob > 0)
        masked_probs = q_is_max_prob[prob_mask]

        entropy = np.zeros_like(q_is_max_prob)
        entropy[prob_mask] = -(masked_probs * np.log(masked_probs) + (1 - masked_probs) * np.log(1 - masked_probs))
        return entropy


class HomoskedasticNoise(object):
    def __init__(self, noise_variance: float):
        self.noise_variance = noise_variance

    def __call__(self, query_points):
        return self.noise_variance * np.ones(*query_points.shape)


class LinearCost(object):
    def __init__(self, base_cost):
        self.base_cost = base_cost

    def __call__(self, query_fidelity):
        return self.base_cost * query_fidelity