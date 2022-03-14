import math

import numpy as np
import torch

from torch import LongTensor

from gpytorch import lazify
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.lazy import ConstantDiagLazyTensor
from gpytorch.settings import cholesky_jitter
from torch.nn import functional as F

from bo_protein.utils import str_to_tokens


def initialize_var_dist_sgpr(model, train_x, train_y, noise_lb):
    """
        This is only intended for whitened variational distributions and gaussian likelihoods 
        at present.

        \bar m = L^{-1} m
        \bar S = L^{-1} S L^{-T}

        where $LL^T = K_{uu}$.

        Thus, the optimal \bar m, \bar S are given by
        \bar S = L^T (K_{uu} + \sigma^{-2} K_{uv} K_{vu})^{-1} L
        \bar m = \bar S L^{-1} (K_{uv} y \sigma^{-2})
    """
    
    if isinstance(model.model.variational_strategy, IndependentMultitaskVariationalStrategy):
        ind_pts = model.model.variational_strategy.base_variational_strategy.inducing_points
        train_y = train_y.transpose(-1, -2).unsqueeze(-1)
        is_batch_model = True
    else:
        ind_pts = model.model.variational_strategy.inducing_points
        is_batch_model = False

    with cholesky_jitter(1e-4):
        kuu = model.model.covar_module(ind_pts).double()
        kuu_chol = kuu.cholesky()
        kuv = model.model.covar_module(ind_pts, train_x).double()

        # noise = model.likelihood.noise if not is_batch_model else model.likelihood.task_noises.unsqueeze(-1).unsqueeze(-1)

        if hasattr(model.likelihood, 'noise'):
            noise = model.likelihood.noise
        elif hasattr(model.likelihood, 'task_noises'):
            noise = model.likelihood.task_noises.view(-1, 1, 1)
        else:
            raise AttributeError
        noise = noise.clamp(min=noise_lb).double()

        if len(train_y.shape) < len(kuv.shape):
            train_y = train_y.unsqueeze(-1)
        if len(noise.shape) < len(kuv.shape):
            noise = noise.unsqueeze(-1)

        data_term = kuv.matmul(train_y.double()) / noise
        # mean_term = kuu_chol.inv_matmul(data_term)
        if is_batch_model:
            # TODO: clean this up a bit more
            noise_as_lt = ConstantDiagLazyTensor(noise.squeeze(-1), diag_shape=kuv.shape[-1])
            inner_prod = kuv.matmul(noise_as_lt).matmul(kuv.transpose(-1, -2))
            inner_term = inner_prod + kuu
        else:
            inner_term = kuv @ kuv.transpose(-1, -2) / noise + kuu

        s_mat = kuu_chol.transpose(-1, -2).matmul(inner_term.inv_matmul(kuu_chol.evaluate()))
        s_root = lazify(s_mat).cholesky().evaluate()
        # mean_param = s_mat.matmul(mean_term)
        # the expression below is less efficient but probably more stable
        mean_param = kuu_chol.transpose(-1, -2).matmul(inner_term.inv_matmul(data_term))

    mean_param = mean_param.to(train_y)
    s_root = s_root.to(train_y)

    if not is_batch_model:
        model.model.variational_strategy._variational_distribution.variational_mean.data = mean_param.data.detach().squeeze()
        model.model.variational_strategy._variational_distribution.chol_variational_covar.data = s_root.data.detach()
        model.model.variational_strategy.variational_params_initialized.fill_(1)
    else:
        model.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data = mean_param.data.detach().squeeze()
        model.model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.data = s_root.data.detach()
        model.model.variational_strategy.base_variational_strategy.variational_params_initialized.fill_(1)


def sample_tokens(base_tokens, logit_batch, enc_tokenizer, replacement=False, temp=1.):
	logit_batch /= temp
	# don't sample special tokens
	non_viable_idxs = np.array(enc_tokenizer.special_idxs)[None, None, :]
	np.put_along_axis(logit_batch, non_viable_idxs, -1e10, axis=-1)

	if not replacement and base_tokens is not None:
		# don't sample the original tokens
		base_tokens = base_tokens.numpy().astype(int)[..., None]
		np.put_along_axis(logit_batch, base_tokens, -1e10, axis=-1)

	# sample tokens
	token_samples = torch.distributions.Categorical(logits=logit_batch).sample()

	# calculate entropy
	entropy = -(
			F.softmax(logit_batch, dim=-1) * F.log_softmax(logit_batch, dim=-1)
	).sum(-1)

	return token_samples, entropy


def sample_mask(
		token_batch: LongTensor,
		tokenizer,
		mask_ratio: float = 0.125,
		mask_size=None
):
	"""
	Args:
		token_batch: (batch_size, num_tokens)
		tokenizer: only necessary to avoid masking special tokens
		mask_ratio: proportion of tokens to mask
		mask_size: (optional) override mask_ratio with a specific mask size
	Returns:
		mask_idxs: (batch_size, mask_size) np.ndarray of position indexes to mask
	"""
	if mask_size is None:
		mask_size = math.ceil(token_batch.shape[-1] * mask_ratio)

	special_idxs = torch.tensor(tokenizer.special_idxs).view(-1, 1, 1)
	is_non_special = token_batch.ne(special_idxs).prod(dim=0).float()
	mask_weights = is_non_special / is_non_special.sum(dim=-1, keepdims=True)
	mask_idxs = torch.multinomial(mask_weights, mask_size, replacement=False)
	return mask_idxs.numpy()


def evaluate_windows(base_seqs, encoder, mask_size, replacement=True, encoder_obj='mlm'):
	window_mask_idxs = {}
	window_entropy = {}
	window_features = {}

	for idx, seq in enumerate(base_seqs):
		window_mask_idxs[idx] = []
		window_entropy[idx] = []
		window_features[idx] = []
		# avoids evaluating windows corresponding to padding tokens
		tokens = str_to_tokens(np.array([seq]), encoder.tokenizer)
		# assert torch.all(tokens.ne(encoder.tokenizer.padding_idx))  # SELFIES no-op token may trigger
		mask_size = min(mask_size, tokens.shape[-1] - 2)
		offset = np.random.randint(1, mask_size + 1)
		for mask_start in range(offset, tokens.shape[-1] - 1, mask_size):
			if mask_start + mask_size < tokens.shape[-1] - 1:
				mask_idxs = np.arange(mask_start, mask_start + mask_size).reshape(1, -1)
			else:
				mask_stop = tokens.shape[-1] - 1
				mask_idxs = np.arange(mask_stop - mask_size, mask_stop).reshape(1, -1)

			with torch.no_grad():
				masked_inputs = tokens.clone().to(encoder.device)
				np.put_along_axis(masked_inputs, mask_idxs, encoder.tokenizer.masking_idx, axis=1)
				tgt_tok_logits, tgt_mask = encoder.logits_from_tokens(masked_inputs)
				if encoder_obj == 'mlm':
					_, logit_entropy = sample_tokens(
						tokens, tgt_tok_logits, encoder.tokenizer, replacement
					)
					logit_entropy = np.take_along_axis(logit_entropy, mask_idxs, axis=1)
				elif encoder_obj == 'lanmt':
					tgt_tok_idxs, logit_entropy = encoder.sample_tgt_tok_idxs(
						tgt_tok_logits, tgt_mask, temp=1.
					)
				else:
					raise ValueError

			window_mask_idxs[idx].append(mask_idxs.copy())
			window_entropy[idx].append(logit_entropy.mean().item())

	return window_mask_idxs, window_entropy