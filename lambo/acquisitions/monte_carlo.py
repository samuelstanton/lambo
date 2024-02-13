from numpy import array, copy, concatenate
from torch import Tensor
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
)
from botorch.posteriors import GPyTorchPosterior, Posterior, DeterministicPosterior
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import BlockDiagLazyTensor
import torch


# TODO: replace these with the non-mocked versions once botorch #991 comes in
# will need to update to botorch master

class qDiscreteEHVI(qExpectedHypervolumeImprovement):
    def forward(self, X: array) -> Tensor:
        # mocks the qEHVI call
        # assumes that X is an array of shape batch x q rather than a tensor of shape batch x q x d
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        return self._compute_qehvi(samples=samples)

    
class qDiscreteNEHVI(qNoisyExpectedHypervolumeImprovement):
    # TODO: figure out how to remove
    
    def __init__(
        self,
        model,
        ref_point,
        X_baseline,
        sampler = None,
        objective = None,
        constraints = None,
        X_pending = None,
        eta: float = 1e-3,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        **kwargs,
    ):
        model.eval()
        mocked_features = model.get_features(X_baseline, model.bs)
        ref_point = ref_point.to(mocked_features)
        # for string kernels
        if mocked_features.ndim > 2:
            mocked_features = mocked_features[..., 0] # don't let this fail

        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=mocked_features,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            **kwargs
        )
        self.X_baseline_string = X_baseline

    def forward(self, X: array) -> Tensor:
        if isinstance(X, Tensor):
            baseline_X = self._X_baseline
            baseline_X = baseline_X.expand(*X.shape[:-2], -1, -1)
            X_full = torch.cat([baseline_X, X], dim=-2)
            q = X.shape[-2]
        else:
            baseline_X = copy(self.X_baseline_string) # ensure contiguity
            baseline_X.resize(
                baseline_X.shape[:-(X.ndim)] + X.shape[:-1] + baseline_X.shape[-1:]
            )
            X_full = concatenate([baseline_X, X], axis=-1)
            q = X.shape[-1]
        # Note: it is important to compute the full posterior over `(X_baseline, X)``
        # to ensure that we properly sample `f(X)` from the joint distribution `
        # `f(X_baseline, X) ~ P(f | D)` given that we can already fixed the sampled
        # function values for `f(X_baseline)`
        posterior = self.model.posterior(X_full)
        self._set_sampler(q=q, posterior=posterior)
        samples = self.sampler(posterior)[..., -q:, :]
        # add previous nehvi from pending points
        return self._compute_qehvi(samples=samples) + self._prev_nehvi
    
    def _cache_root_decomposition(self, posterior: GPyTorchPosterior) -> None:
        if posterior.mvn._interleaved:
            if hasattr(posterior.mvn.lazy_covariance_matrix, 'base_lazy_tensor'):
                posterior_lc_base = posterior.mvn.lazy_covariance_matrix.base_lazy_tensor
            else:
                posterior_lc_base = posterior.mvn.lazy_covariance_matrix

            new_lazy_covariance = BlockDiagLazyTensor(posterior_lc_base)
            posterior.mvn = MultitaskMultivariateNormal(posterior.mvn.mean, new_lazy_covariance, interleaved=False)
        return super()._cache_root_decomposition(posterior=posterior)
    
    
class qMTGPDiscreteNEHVI(qDiscreteNEHVI):
    # TODO: remove when botorch #1037 goes in
    # this is copied over from that diff
    
    _uses_matheron = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(cache_root = False, *args, **kwargs)
    
    def _set_sampler(
        self,
        q: int,
        posterior: Posterior,
    ) -> None:
        r"""Update the sampler to use the original base samples for X_baseline.
        Args:
            q: the batch size
            posterior: the posterior
        TODO: refactor some/all of this into the MCSampler.
        """
        if self.q != q:
            # create new base_samples
            base_sample_shape = self.sampler._get_base_sample_shape(posterior=posterior)
            self.sampler._construct_base_samples(
                posterior=posterior, shape=base_sample_shape
            )
            if (
                self.X_baseline.shape[0] > 0
                and self.base_sampler.base_samples is not None
                and not isinstance(posterior, DeterministicPosterior)
            ):
                current_base_samples = self.base_sampler.base_samples.detach().clone()
                # This is the # of non-`sample_shape` dimensions.
                base_ndims = current_base_samples.dim() - 1
                # Unsqueeze as many dimensions as needed to match base_sample_shape.
                view_shape = (
                    self.sampler.sample_shape
                    + torch.Size(
                        [1] * (len(base_sample_shape) - current_base_samples.dim())
                    )
                    + current_base_samples.shape[-base_ndims:]
                )
                expanded_shape = (
                    base_sample_shape[:-base_ndims]
                    + current_base_samples.shape[-base_ndims:]
                )
                # Use stored base samples:
                # Use all base_samples from the current sampler
                # this includes the base_samples from the base_sampler
                # and any base_samples for the new points in the sampler.
                # For example, when using sequential greedy candidate generation
                # then generate the new candidate point using last (-1) base_sample
                # in sampler. This copies that base sample.
                end_idx = current_base_samples.shape[-1 if self._uses_matheron else -2]
                expanded_samples = current_base_samples.view(view_shape).expand(
                    expanded_shape
                )
                if self._uses_matheron:
                    self.sampler.base_samples[..., :end_idx] = expanded_samples
                else:
                    self.sampler.base_samples[..., :end_idx, :] = expanded_samples
                # update cached subset indices
                # Note: this also stores self.q = q
                self._cache_q_subset_indices(q=q)
