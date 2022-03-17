import torch
import numpy as np

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors import GPyTorchPosterior
from gpytorch import settings
# from gpytorch.lazy import BatchRepeatLazyTensor
from botorch.models import SingleTaskGP

from ..utils import AMINO_ACIDS, Expression
from .sskernel import SSKernel
from .gp_models import BaseGPSurrogate, SingleTaskExactGP
from .gp_utils import fit_gp_surrogate
from bo_protein.transforms import padding_collate_fn
from bo_protein import dataset as dataset_util


class SSKExactGP(SingleTaskExactGP):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    def __init__(
        self, 
        out_dim, 
        max_depth,
        gap_decay,
        str_max_len, 
        alphabet=None,
        likelihood=None, 
        tokenizer=None,
        encoder=None,
        outcome_transform=None, 
        task_noise_init=0.5,
        eval_bs=20,
        num_splits=1,
        *args, 
        **kwargs
    ):
        encoder = Expression(lambda x: x)
        # initialize common attributes
        BaseGPSurrogate.__init__(self, tokenizer=tokenizer, encoder=encoder, *args, **kwargs)

        # initialize GP
        dummy_X = torch.randn(2, 3).to(self.device)
        dummy_Y = torch.randn(2, out_dim).to(self.device)

        covar_module = SSKernel(
            max_depth=max_depth, gap_decay=gap_decay, num_splits=num_splits
        ).to(self.device)
        SingleTaskGP.__init__(
            self, dummy_X, dummy_Y, likelihood, covar_module, outcome_transform, input_transform=None,
        )

        self.likelihood.initialize(noise=task_noise_init)
        
        if alphabet is None:
            alphabet = AMINO_ACIDS
        self.alphabet = alphabet
        self.str_max_len = str_max_len

        self.prediction_cache = None
        self.tokenizer = tokenizer
        self.eval_bs = eval_bs
        self.encoder = encoder
        self.to(self.device)

#     def prepare_match_features(self, all_preprocessed_strs):
#         alph_range = range(len(self.alphabet) + 1) # zero padding + alphabet length
#         match_inputs_list = []
#         for split_str in all_preprocessed_strs:
#             match_inputs_list.append(
#                 torch.stack([split_str == alph for alph in alph_range]).T
#             )
#         inputs_as_match = torch.stack(match_inputs_list)
#         return inputs_as_match.to(self.device)

    def prepare_match_features(self, all_preprocessed_strs):
        num_input_dims = len(all_preprocessed_strs.shape)
        device = all_preprocessed_strs.device
        valid_tok_idxs = torch.tensor(self.tokenizer.non_special_idxs, device=device)
        view_dims = [1] * num_input_dims
        valid_tok_idxs = valid_tok_idxs.view(-1, *view_dims)
        matches = valid_tok_idxs.eq(all_preprocessed_strs)  # (len(alphabet), *batch_shape, num_tokens)
        permute_dims = list(range(1, num_input_dims + 1))
        return matches.permute(*permute_dims, 0).to(self.device)  # (*batch_shape, num_tokens, len(alphabet))
    
    def get_features(self, seq_array, batch_size=None, transform=True):
        """
        moves from an array of sequences to a q x n x L x |a| tensor of binary matches
        """
        if seq_array.ndim == 1:
            if isinstance(seq_array, np.ndarray):
                tokens = [
                    torch.tensor(self.tokenizer.encode(x)) for x in seq_array
                ]
                all_preprocessed_strs = padding_collate_fn(tokens, self.tokenizer.padding_idx)
                if all_preprocessed_strs.shape[-1] > self.str_max_len:
                    all_preprocessed_strs = all_preprocessed_strs[..., :self.str_max_len]
                elif all_preprocessed_strs.shape[-1] < self.str_max_len:
                    padding = torch.zeros(
                        all_preprocessed_strs.shape[-2], self.str_max_len - all_preprocessed_strs.shape[-1],
                    ).to(all_preprocessed_strs)
                    all_preprocessed_strs = torch.cat((all_preprocessed_strs, padding), -1)
            else:
                if seq_array.dtype is torch.bool:
                    return seq_array
                else:
                    all_preprocessed_strs = seq_array
            return self.prepare_match_features(all_preprocessed_strs)
        else:
            seq_list = []
            for seq in seq_array:
                seq_list.append(self.get_features(seq, batch_size=batch_size))
            res = torch.stack(seq_list).to(self.device)
            return res

    def forward(self, inputs):
        features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
        features = features.to(self.device)
        covar = self.covar_module(features)
        mean = self.mean_module(torch.empty(features.shape[:-2], device=features.device))
        return MultivariateNormal(mean, covar)

    def __call__(self, *args, **kwds):
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        if self.training:
            return super().__call__(*args, **kwds)
        else:
            train_features = self.train_inputs[0]
            
            if self.prediction_cache is None:
                with torch.no_grad():
                    self.train_targets = self.train_targets.to(list(self.covar_module.parameters())[0])
                    
                    # need to manually put the posterior together        
                    train_train_covar = self.covar_module(train_features)
                    train_mean = self.mean_module(torch.empty(train_features.shape[:-2], device=train_features.device))
                    train_with_likelihood = self.likelihood(MultivariateNormal(train_mean, train_train_covar))

                    train_diff = self.train_targets.squeeze() - train_mean
                    covar_with_noise = train_with_likelihood.lazy_covariance_matrix
                    train_cache = covar_with_noise.inv_matmul(train_diff.unsqueeze(-1))
                    with settings.max_cholesky_size(50000):
                        covar_inv_root = covar_with_noise.root_inv_decomposition().root
                    self.prediction_cache = (train_cache, covar_inv_root)

            test_features = inputs[0].squeeze()
            if test_features.dtype is not torch.bool:
                test_features = self.prepare_match_features(test_features)
                
            if test_features.shape[-2] < train_features.shape[-2]:
                zero_padding_dim = torch.zeros(
                    *test_features.shape[:-2], 
                    train_features.shape[-2] - test_features.shape[-2],
                    train_features.shape[-1],
                ).to(test_features)
                test_features = torch.cat((test_features, zero_padding_dim), dim=-2)
            elif test_features.shape[-2] > self.str_max_len:
                test_features = test_features[..., :self.str_max_len, :]
                
            test_test_covar = self.covar_module(test_features)
            test_train_covar = self.covar_module(test_features, train_features).evaluate()
            # now fix the batch dimensions as necessary

            mean_cache, covar_cache = self.prediction_cache
            # print("shapes, ", test_test_covar.shape, test_train_covar.shape, mean_cache.shape, covar_cache.shape)
            covar_cache = covar_cache.evaluate()
            had_double_batch = False
            if mean_cache.shape[:-2] != test_train_covar.shape[:-2]:
                # in this case, we need to match the batch shapes
                test_train_covar = test_train_covar.unsqueeze(0).repeat(
                    self._num_outputs, *[1]*len(test_train_covar.shape[:-1]), 1
                )
                if mean_cache.ndim < test_train_covar.ndim:
                    had_double_batch = True
                    while mean_cache.ndim < test_train_covar.ndim:
                        mean_cache = mean_cache.unsqueeze(-3)
                        covar_cache = covar_cache.unsqueeze(-3)
                    mean_cache = mean_cache.repeat(1, test_train_covar.shape[-3], 1, 1)
                    covar_cache = covar_cache.repeat(1, test_train_covar.shape[-3], 1, 1)


            # something with 1 predictor here
            if test_features.ndim < 3: 
                test_features = test_features.unsqueeze(0)
                
            test_mean = self.mean_module(torch.empty(
                *test_features.shape[:-2], device=train_features.device
            ))
            if had_double_batch:
                # TODO: why does this not cut across the 2nd q batch dim
                test_mean = test_mean.unsqueeze(-1).repeat(*[1]*test_mean.ndim, test_features.shape[-3])
 
            pred_mean = test_train_covar.matmul(mean_cache) + test_mean.unsqueeze(-1)
          
            second_term = test_train_covar.matmul(covar_cache)
            # print(test_train_covar.matmul(mean_cache).shape)
            pred_covar = test_test_covar - second_term.matmul(second_term.transpose(-1, -2))

            pred_mean = pred_mean.squeeze(-1)
            return MultivariateNormal(pred_mean, pred_covar)

    def train(self, *args, **kwargs):
        self.prediction_cache = None
        return super().train(*args, **kwargs)

    def _posterior(self, X, output_indices = None, observation_noise= False, **kwargs,):
        X = self.get_features(X, self.bs) if isinstance(X, np.ndarray) else X

        # this is copied over from botorch
        # https://github.com/pytorch/botorch/blob/8ceae970c07bf684382769758c051f39c8045683/botorch/models/gpytorch.py#L202
        # with a slight modification for some reason
        self.eval()
        mvn = self(X)
        if observation_noise is not False:
            if torch.is_tensor(observation_noise):
                # TODO: Validate noise shape
                # make observation_noise `batch_shape x q x n`
                obs_noise = observation_noise.transpose(-1, -2)
                mvn = self.likelihood(mvn, X, noise=obs_noise)
            elif isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
                # Use the mean of the previous noise values (TODO: be smarter here).
                noise = self.likelihood.noise.mean().expand(X.shape[:-1])
                mvn = self.likelihood(mvn, X, noise=noise)
            else:
                mvn = self.likelihood(mvn, X)
        if self._num_outputs > 1:
            mean_x = mvn.mean
            covar_x = mvn.lazy_covariance_matrix
            output_indices = output_indices or range(self._num_outputs)
            mvns = [
                MultivariateNormal(
                    mean_x[t],
                    covar_x[t],
                )
                for t in output_indices
            ]
            mvn = MultitaskMultivariateNormal.from_independent_mvns(mvns=mvns)

        posterior = GPyTorchPosterior(mvn=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior
    
    def posterior(self, X, output_indices = None, observation_noise= False, batch_size=None, **kwargs,):
        if (batch_size is None and self.eval_bs is None) or X.shape[-1] < self.eval_bs or X.shape[-1] < batch_size:
            return self._posterior(X=X, output_indices=output_indices, observation_noise=observation_noise, **kwargs)
        elif batch_size is None:
            batch_size = self.eval_bs
        
        # else we need to construct a dataloader
        self.eval()
        collate_fn = padding_collate_fn
        
        Y_test = torch.zeros(*X.shape) # mock to ensure we properly setup the dataset
        test_dataset = dataset_util.TransformTensorDataset(
            [X, Y_test], self.test_transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size = batch_size, shuffle=False, collate_fn=collate_fn
        )
        dist_list = []
        for input_batch, target_batch in test_loader:
            features = self.encoder(input_batch.to(self.device))
            post = self._posterior(
                features, output_indices=output_indices, observation_noise=observation_noise, **kwargs
            )
            dist_list.append(post.mvn)
            
        concatenated_mean = torch.cat([dist.mean.squeeze() for dist in dist_list], dim=-1)
        # form the non-square block diagonal matrix
        if concatenated_mean.ndim > 1:
            concat_covar = torch.zeros(
                *concatenated_mean.shape[:-1], concatenated_mean.shape[-1], concatenated_mean.shape[-1], 
                device = concatenated_mean.device, dtype = concatenated_mean.dtype,
            )
        else:
            concat_covar = torch.zeros(
                concatenated_mean.shape[-1], concatenated_mean.shape[-1], 
                device = concatenated_mean.device, dtype = concatenated_mean.dtype,
            )
        end_ind = 0
        start_ind = 0
        for i, dist in enumerate(dist_list):
            if i > 0:
                start_ind = end_ind
            covar = dist.covariance_matrix
            end_ind = covar.shape[-2] + end_ind
            concat_covar[..., start_ind:end_ind, start_ind:end_ind] = covar
            
        dist_type = MultivariateNormal if isinstance(dist_list[0], MultivariateNormal) else MultitaskMultivariateNormal
        concat_dist = dist_type(concatenated_mean, concat_covar)
        return GPyTorchPosterior(concat_dist)

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, reset=False, log_prefix="single_task_string_kernel",
            **kwargs):
        if reset:
            raise NotImplementedError
            
        # set train data
        if Y_train.shape[-1] > 1:
            Y_tsr = torch.tensor(Y_train).float().t()
        else:
            Y_tsr = torch.tensor(Y_train).float()
        self.set_train_data(X_train, Y_tsr, strict=False)
        
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        
        return fit_gp_surrogate(
            surrogate=self,
            mll=mll,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            eval_bs=self.eval_bs,
            train_bs=X_train.shape[0],
            encoder_obj=None,
            log_prefix=log_prefix
        )

    def set_train_data(self, inputs=None, targets=None, strict=True):
        train_features = self.get_features(inputs, self.bs) if isinstance(inputs, np.ndarray) else inputs
        targets = targets.to(self.device)
        SingleTaskGP.set_train_data(self, train_features, targets, strict)
        
    def clear_cache(self):
        self.train()
        self.prediction_cache = None
