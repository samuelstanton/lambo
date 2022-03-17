import torch
import numpy as np
import abc

from scipy.stats import spearmanr

from botorch.models import SingleTaskGP, SingleTaskVariationalGP, KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch import likelihoods, kernels

from bo_protein.models.base_surrogate import BaseSurrogate
from bo_protein import transforms as gfp_transforms
from bo_protein.models.metrics import quantile_calibration

from .gp_utils import fit_gp_surrogate


class BaseGPSurrogate(BaseSurrogate, abc.ABC):
    def __init__(self, max_shift, mask_size, gp_lr, enc_lr, bs, num_epochs, holdout_ratio, early_stopping,
                 patience, eval_period, tokenizer, encoder, encoder_wd=0., bootstrap_ratio=None, min_num_train=128,
                 task_noise_init=0.01, lengthscale_init=0.7, *args, **kwargs):
        self.gp_lr = gp_lr
        self.enc_lr = enc_lr
        self.bs = bs
        self.encoder_wd = encoder_wd
        self.num_epochs = num_epochs
        self.holdout_ratio = holdout_ratio
        self.early_stopping = early_stopping
        self.patience = patience
        self.eval_period = eval_period
        self.bootstrap_ratio = bootstrap_ratio
        self.min_num_train = min_num_train
        self.task_noise_init = task_noise_init
        self.lengthscale_init = lengthscale_init
        self.tokenizer = tokenizer

        self._set_transforms(tokenizer, max_shift, mask_size)

    def get_features(self, seq_array, batch_size=None, transform=True):
        if transform:
            original_shape = seq_array.shape
            flat_seq_array = seq_array.reshape(-1)
        else:
            original_shape = seq_array.shape[:-1]
            flat_seq_array = seq_array.flatten(end_dim=-2)

        if self.training and transform:
            enc_seq_array = gfp_transforms.padding_collate_fn(
                [self.train_transform(seq) for seq in flat_seq_array],
                self.tokenizer.padding_idx,
            )
        elif transform:
            enc_seq_array = gfp_transforms.padding_collate_fn(
                [self.test_transform(seq) for seq in flat_seq_array],
                self.tokenizer.padding_idx,
            )
        else:
            enc_seq_array = seq_array

        enc_seq_array = enc_seq_array.to(self.device)
        features = self.encoder(enc_seq_array)

        return features.view(*original_shape, -1)

    def reshape_targets(self, targets):
        return targets

    def predict(self, inputs, num_samples=1, latent=False):
        self.eval()
        with torch.inference_mode():
            features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
            pred_dist = self(features)
            pred_dist = pred_dist if latent else self.likelihood(pred_dist)

        pred_mean = pred_dist.mean.cpu()
        pred_std = pred_dist.variance.sqrt().cpu()
        batch_shape = features.shape[:-1]

        if pred_mean.ndim == 1:
            pred_mean = pred_mean.unsqueeze(-1)
            pred_std = pred_std.unsqueeze(-1)
        elif not pred_mean.shape[:-1] == batch_shape:
            pred_mean = pred_mean.transpose(-1, -2)
            pred_std = pred_mean.transpose(-1, -2)
        else:
            pass
        assert pred_mean.shape[:-1] == batch_shape, f'{pred_mean.shape[:-1]} != {batch_shape}'

        diag_dist = torch.distributions.Normal(pred_mean, pred_std)
        samples = diag_dist.sample((num_samples,))
        return samples, pred_mean, pred_std

    def evaluate(self, loader, split="", *args, **kwargs):
        self.eval()
        targets, y_mean, y_std, f_std = [], [], [], []
        with torch.no_grad():
            for input_batch, target_batch in loader:
                # features = self.get_features(input_batch.to(self.device), self.bs, transform=False)
                features = self.get_features(input_batch.to(self.device), transform=False)
                f_dist = self(features)
                y_dist = self.likelihood(f_dist)

                target_batch = self.reshape_targets(target_batch)
                targets.append(target_batch.to(features.device).cpu())
                # import pdb; pdb.set_trace()
                if y_dist.mean.shape == target_batch.shape:
                    f_std.append(f_dist.variance.sqrt().cpu())
                    y_mean.append(y_dist.mean.cpu())
                    y_std.append(y_dist.variance.sqrt().cpu())
                else:
                    f_std.append(
                        f_dist.variance.sqrt().cpu().transpose(-1, -2)
                    )
                    y_mean.append(
                        y_dist.mean.cpu().transpose(-1, -2)
                    )
                    y_std.append(
                        y_dist.variance.sqrt().cpu().transpose(-1, -2)
                    )

        # TODO: figure out why these are getting flipped
        try:
            targets = torch.cat(targets).view(len(loader.dataset), -1)
            cat_dim = 0
        except:
            targets = torch.cat(targets, -1).view(len(loader.dataset), -1)
            cat_dim = -1
        f_std = torch.cat(f_std, cat_dim).view(len(loader.dataset), -1)
        y_mean = torch.cat(y_mean, cat_dim).view(len(loader.dataset), -1)
        y_std = torch.cat(y_std, cat_dim).view(len(loader.dataset), -1)

        assert y_mean.shape == targets.shape

        rmse = (y_mean - targets).pow(2).mean().sqrt()
        nll = -torch.distributions.Normal(y_mean, y_std).log_prob(targets).mean()
        cal_metrics = quantile_calibration(y_mean, y_std, targets)
        ece = cal_metrics["ece"]
        occ_diff = cal_metrics["occ_diff"]

        spearman_rho = 0
        for idx in range(targets.size(-1)):
            spearman_rho += spearmanr(targets[..., idx], y_mean[..., idx]).correlation / targets.size(-1)

        metrics = {
            f"{split}_nll": nll.item(),
            f"{split}_rmse": rmse.item(),
            f"{split}_s_rho": spearman_rho,
            f"{split}_ece": ece,
            f"{split}_occ_diff": occ_diff,
            f"{split}_post_var": (f_std ** 2).mean().item()
        }

        if hasattr(self.likelihood, 'task_noises'):
            metrics['noise'] = self.likelihood.task_noises.mean().item()
        elif hasattr(self.likelihood, 'noise'):
            metrics['noise'] = self.likelihood.noise.mean().item()
        else:
            pass

        covar_module = self.model.covar_module if hasattr(self, 'model') else self.covar_module
        if hasattr(covar_module, 'base_kernel') and hasattr(covar_module.base_kernel, 'lengthscale'):
            metrics['lengthscale'] = covar_module.base_kernel.lengthscale.mean().item()
        elif hasattr(covar_module, 'data_covar_module'):
            metrics['lengthscale'] = covar_module.data_covar_module.lengthscale.mean().item()
        elif hasattr(covar_module, 'lengthscale'):
            metrics['lengthscale'] = covar_module.lengthscale.mean().item()
        else:
            pass

        if hasattr(covar_module, 'outputscale'):
            metrics['outputscale'] = covar_module.outputscale.mean().item()

        return metrics

    @property
    def param_groups(self):

        gp_hypers = dict(params=[], lr=self.gp_lr)
        noise_group = dict(params=[], lr=self.gp_lr)
        inducing_point_group = dict(params=[], lr=self.gp_lr)
        variational_group = dict(params=[], lr=self.gp_lr)

        for name, param in self.named_parameters():
            if name.split('.')[0] == 'encoder':
                continue
            if 'noise' in name:
                noise_group['params'].append(param)
            elif 'inducing_points' in name:
                inducing_point_group['params'].append(param)
            elif 'variational_distribution' in name:
                variational_group['params'].append(param)
            else:
                gp_hypers['params'].append(param)

        param_groups = [gp_hypers]

        if hasattr(self, "encoder") and hasattr(self.encoder, 'param_groups'):
            param_groups.extend(self.encoder.param_groups(self.enc_lr, self.encoder_wd))

        if len(noise_group['params']) > 0:
            param_groups.append(noise_group)

        if len(inducing_point_group['params']) > 0:
            param_groups.append(inducing_point_group)

        if len(variational_group['params']) > 0:
            param_groups.append(variational_group)

        return param_groups


class SingleTaskExactGP(BaseGPSurrogate, SingleTaskGP):
    def __init__(self, feature_dim, out_dim, encoder, likelihood=None, covar_module=None,
                 outcome_transform=None, input_transform=None, *args, **kwargs):

        # initialize common attributes
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)

        # initialize GP
        dummy_X = torch.randn(2, feature_dim).to(self.device)
        dummy_Y = torch.randn(2, out_dim).to(self.device)
        covar_module = covar_module if covar_module is None else covar_module.to(self.device)
        SingleTaskGP.__init__(
            self, dummy_X, dummy_Y, likelihood, covar_module, outcome_transform, input_transform
        )
        self.likelihood.initialize(noise=self.task_noise_init)
        self.encoder = encoder.to(self.device)

    def clear_cache(self):
        self.train()
        
    def forward(self, inputs):
        features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
        return SingleTaskGP.forward(self, features)

    def posterior(self, inputs, output_indices=None, observation_noise=False, **kwargs):
        features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
        return SingleTaskGP.posterior(self, features, output_indices, observation_noise, **kwargs)

    def reshape_targets(self, targets):
        return targets.transpose(-1, -2)

    def set_train_data(self, inputs=None, targets=None, strict=True):
        train_features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
        SingleTaskGP.set_train_data(self, train_features, targets.to(train_features), strict)

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, reset=False, log_prefix="single_task_gp", **kwargs):
        if reset:
            raise NotImplementedError
        fit_kwargs = dict(
            surrogate=self,
            mll=ExactMarginalLogLikelihood(self.likelihood, self),
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            train_bs=None,
            eval_bs=self.bs,
            shuffle_train=False,
            log_prefix=log_prefix
        )
        return fit_gp_surrogate(**fit_kwargs, **kwargs)


class MultiTaskExactGP(BaseGPSurrogate, KroneckerMultiTaskGP):
    def __init__(self, feature_dim, out_dim, encoder, likelihood=None, covar_module=None,
                 outcome_transform=None, input_transform=None, *args, **kwargs):

        # initialize common attributes
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)

        # initialize GP
        dummy_X = torch.randn(2, feature_dim).to(self.device)
        dummy_Y = torch.randn(2, out_dim).to(self.device)
        covar_module = covar_module if covar_module is None else covar_module.to(self.device)
        KroneckerMultiTaskGP.__init__(
            self, dummy_X, dummy_Y, likelihood, covar_module=covar_module, outcome_transform=outcome_transform,
            input_transform=input_transform, *args, **kwargs
        )
        self.likelihood.initialize(task_noises=self.task_noise_init)
        self.encoder = encoder.to(self.device)

    def forward(self, X):
        features = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X
        return KroneckerMultiTaskGP.forward(self, features)

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        features = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X
        return KroneckerMultiTaskGP.posterior(self, features, output_indices, observation_noise, **kwargs)

    def clear_cache(self):
        clear_cache_hook(self)
        self.prediction_strategy = None

    def set_train_data(self, X=None, targets=None, strict=True):
        self.clear_cache()
        train_features = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X
        KroneckerMultiTaskGP.set_train_data(self, train_features, targets.to(train_features), strict)

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, reset=False, log_prefix="multi_task_gp", **kwargs):
        if reset:
            raise NotImplementedError
        fit_kwargs = dict(
            surrogate=self,
            mll=ExactMarginalLogLikelihood(self.likelihood, self),
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            train_bs=None,
            eval_bs=self.bs,
            shuffle_train=True,
            log_prefix=log_prefix
        )
        fit_kwargs.update(kwargs)
        return fit_gp_surrogate(**fit_kwargs)


class SingleTaskSVGP(BaseGPSurrogate, SingleTaskVariationalGP):
    def __init__(self, feature_dim, out_dim, num_inducing_points, encoder, noise_constraint=None, lengthscale_prior=None,
                 outcome_transform=None, input_transform=None, learn_inducing_points=True, mll_beta=1.,
                 *args, **kwargs
                 ):

        # initialize common attributes
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)
        self.num_inducing_points = num_inducing_points

        if out_dim == 1:
            covar_module = kernels.MaternKernel(
                ard_num_dims=feature_dim, lengthscale_prior=lengthscale_prior
            )
            covar_module.initialize(lengthscale=self.lengthscale_init)
            likelihood = likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint
            )
            likelihood.initialize(noise=self.task_noise_init)
        else:
            covar_module = kernels.MaternKernel(
                batch_shape=(out_dim,), ard_num_dims=feature_dim, lengthscale_prior=lengthscale_prior
            )
            covar_module.initialize(lengthscale=self.lengthscale_init)
            likelihood = likelihoods.MultitaskGaussianLikelihood(
                num_tasks=out_dim, has_global_noise=False, noise_constraint=noise_constraint
            )
            likelihood.initialize(task_noises=self.task_noise_init)

        # initialize GP
        dummy_X = 2 * (torch.rand(num_inducing_points, feature_dim).to(self.device) - 0.5)
        dummy_Y = torch.randn(num_inducing_points, out_dim).to(self.device)
        covar_module = covar_module if covar_module is None else covar_module.to(self.device)

        self.base_cls = SingleTaskVariationalGP
        self.base_cls.__init__(self, dummy_X, dummy_Y, likelihood, out_dim, learn_inducing_points,
                                         covar_module=covar_module, inducing_points=dummy_X,
                                         outcome_transform=outcome_transform, input_transform=input_transform)
        self.encoder = encoder.to(self.device)
        self.mll_beta = mll_beta

    def clear_cache(self):
        clear_cache_hook(self)
        clear_cache_hook(self.model)
        clear_cache_hook(self.model.variational_strategy)
        if hasattr(self.model.variational_strategy, 'base_variational_strategy'):
            clear_cache_hook(self.model.variational_strategy.base_variational_strategy)

    def forward(self, inputs):
        features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
        res = self.base_cls.forward(self, features)
        return res

    def posterior(self, inputs, output_indices=None, observation_noise=False, **kwargs):
        self.clear_cache()
        features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
        return self.base_cls.posterior(self, features, output_indices, observation_noise, **kwargs)

    def set_train_data(self, inputs=None, targets=None, strict=True):
        self.clear_cache()

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, reset=False, log_prefix="single_task_svgp", **kwargs):
        if reset:
            raise NotImplementedError

        fit_kwargs = dict(
            surrogate=self,
            mll=VariationalELBO(self.likelihood, self.model, num_data=X_train.shape[0]),
            # mll=PredictiveLogLikelihood(
            #     self.likelihood, self.model, num_data=X_train.shape[0], beta=self.mll_beta
            # ),
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            train_bs=self.bs,
            eval_bs=self.bs,
            shuffle_train=True,
            log_prefix=log_prefix
        )
        fit_kwargs.update(kwargs)
        return fit_gp_surrogate(**fit_kwargs)

    def reshape_targets(self, targets):
        if targets.shape[-1] > 1:
            return targets
        else:
            return targets.squeeze(-1)
