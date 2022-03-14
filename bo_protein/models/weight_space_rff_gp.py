#!/usr/bin/env python3

from typing import Optional, Union, List, Any

import gpytorch
import torch

from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.transforms.input import InputTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.utils import (
    validate_input_scaling, 
    gpt_posterior_settings,
    add_output_dim,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import RFFKernel
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, Likelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.priors import GammaPrior, Prior
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.utils.memoize import cached, clear_cache_hook
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    _VariationalDistribution,
    _VariationalStrategy,
)
from torch import Size, Tensor
from torch.nn import Module


class _WeightSpaceVariationalStrategy(_VariationalStrategy):
    def __init__(
        self,
        model: ApproximateGP,
        variational_distribution: _VariationalDistribution,
        prior_variance: float = 1.0,
    ):
        super().__init__(
            model=model,
            variational_distribution=variational_distribution,
            inducing_points=torch.tensor([]),
            learn_inducing_locations=False,
        )
        self.prior_variance = prior_variance

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        mean = torch.zeros_like(self._variational_distribution().mean)
        var = self.prior_variance * torch.ones_like(mean)
        return MultivariateNormal(mean, DiagLazyTensor(var))

    def kl_divergence(self) -> Tensor:
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.
        :rtype: torch.Tensor
        """
        # gpytorch internals enforce no CG here, requiring cholesky.
        # strategy keeps the same setting
        with gpytorch.settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(
                self._variational_distribution(), self.prior_distribution
            )
        return kl_divergence


class _RFFWeightSpaceModel(ApproximateGP):
    def __init__(
        self,
        mean_module: Optional[Module] = None,
        num_samples: int = 100,
        prior_variance: float = 1.0,
        batch_shape: Size = None,
        train_X: Tensor = None,
        train_Y: Tensor = None,
        num_dims: int = None,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Interval] = None,
        # input_transform: Optional[InputTransform] = None,
        # outcome_transform: Optional[OutcomeTransform] = None,
    ):
        r"""
        Weight space variational bayesian inference for Random fourier features GPs.

        Args:
            feature_extractor: A neural network architecture for joint training.
            mean_module: mean function for the GP. Default is ConstantMean.
            num_samples: number of random fourier features to use.
            prior_variance: variance of weight space prior.
            batch_shape: batch shaping (e.g. for independent multi-outputs).
        """
        if batch_shape is None:
            self.batch_shape = Size()
        else:
            self.batch_shape = batch_shape

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=2 * num_samples, batch_shape=self.batch_shape
        )
        variational_strategy = _WeightSpaceVariationalStrategy(
            model=self,
            variational_distribution=variational_distribution,
            prior_variance=prior_variance,
        )
        super().__init__(variational_strategy=variational_strategy)
        if mean_module is None:
            self.mean_module = ConstantMean(batch_shape=self.batch_shape)

        # self.input_transform = input_transform

        input_dict = {
            "num_samples": num_samples, 
            "batch_shape": self.batch_shape,
            "lengthscale_prior": lengthscale_prior,
            "lengthscale_constraint": lengthscale_constraint,
        }

        if train_X is not None:
            self.train_inputs = [train_X]
            input_dict["ard_num_dims"] = train_X.shape[-1]

        if train_Y is not None:
            self.train_targets = train_Y

        if num_dims is not None:
            input_dict["ard_num_dims"] = num_dims

        self.covar_module = RFFKernel(**input_dict)

    def __call__(self, x: Tensor, prior: bool = False) -> MultivariateNormal:
        if prior:
            return super().__call__(x, prior=prior)
        else:
            if self.training:
                clear_cache_hook(self)

            return self.forward(x)

    def forward(self, X: Tensor) -> MultivariateNormal:
        # if self.input_transform is not None:
        #     X = self.input_transform(X)

        var_dist = self.variational_strategy._variational_distribution()
        feature_covariance = self.covar_module(X).evaluate_kernel()
        if isinstance(feature_covariance, RootLazyTensor):
            features = feature_covariance.root
        elif isinstance(feature_covariance, MatmulLazyTensor):
            features = feature_covariance.left_lazy_tensor
        else:
            raise RuntimeError(
                f"Cannot interpret feature covariance of type {type(feature_covariance)}."
            )

        # mean function is X\beta + \mu(inputs)
        var_mean = var_dist.mean
        # in multi-batch case, unsqueeze x for the setting
        gp_mean = self.mean_module(X)
        if self.batch_shape == Size():
            var_mean = var_mean.unsqueeze(0)
            gp_mean = gp_mean.unsqueeze(0)

        first_term = features.matmul(var_mean.unsqueeze(-1)).squeeze()
        mean_function = first_term + gp_mean

        if self.batch_shape == Size():
            mean_function = mean_function.squeeze()

        covar_function = RootLazyTensor(
            features.matmul(var_dist.lazy_covariance_matrix.root)
        )
        return MultivariateNormal(mean_function, covar_function)


class RFFWeightSpaceGP(ApproximateGPyTorchModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        num_samples: int = 100,
        prior_variance: float = 1.0,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Interval] = None,
    ) -> None:
        r"""
        A single task variational RFF GP. The variational component allows mini-batching.

        Args:
            TODO
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if train_Y is not None:
            if outcome_transform is not None:
                train_Y, _ = outcome_transform(train_Y)
            self._validate_tensor_args(X=transformed_X, Y=train_Y)
            validate_input_scaling(train_X=transformed_X, train_Y=train_Y)
            if train_Y.shape[-1] != num_outputs:
                num_outputs = train_Y.shape[-1]

        self._num_outputs = num_outputs
        self._input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = self._input_batch_shape
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        self._aug_batch_shape = aug_batch_shape

        if likelihood is None:
            if num_outputs == 1:
                noise_prior = GammaPrior(1.1, 0.05)
                noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
                likelihood = GaussianLikelihood(
                    noise_prior=noise_prior,
                    batch_shape=self._aug_batch_shape,
                    noise_constraint=GreaterThan(
                        1e-4,
                        # transform=None,
                        initial_value=noise_prior_mode,
                    ),
                )
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self._is_custom_likelihood = True

        model = _RFFWeightSpaceModel(
            train_X=transformed_X,
            train_Y=train_Y,
            # num_outputs=num_outputs,
            mean_module=mean_module,
            num_samples=num_samples,
            prior_variance=prior_variance,
            batch_shape=torch.Size((train_Y.shape[-1],)),
            num_dims=transformed_X.shape[-1],
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

        super().__init__(model=model, likelihood=likelihood, num_outputs=num_outputs)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

        # for model fitting utilities
        # TODO: make this a flag?
        self.model.train_inputs = [transformed_X]
        if train_Y is not None:
            self.model.train_targets = train_Y.squeeze(-1)

        self.to(train_X)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        return BatchedMultiOutputGPyTorchModel.posterior(
            self=self, X=X, output_indices=output_indices, observation_noise=observation_noise, **kwargs
        )
    