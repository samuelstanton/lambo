import math

import numpy as np
import torch
import torchvision
import wandb

from torch.nn import functional as F
from torch import LongTensor

from lambo import transforms as gfp_transforms, dataset as gfp_dataset
from lambo.models.shared_elements import check_early_stopping
from lambo.utils import str_to_tokens


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


def mlm_train_step(model, optimizer, token_batch, mask_ratio, loss_scale=1.):
	optimizer.zero_grad(set_to_none=True)

	# replace random tokens with mask token
	mask_idxs = sample_mask(token_batch, model.tokenizer, mask_ratio)
	masked_token_batch = token_batch.clone().to(model.device)
	np.put_along_axis(masked_token_batch, mask_idxs, model.tokenizer.masking_idx, axis=1)

	# get predicted logits for masked tokens
	logits, _ = model.logits_from_tokens(masked_token_batch)
	vocab_size = logits.shape[-1]
	masked_logits = np.take_along_axis(logits, mask_idxs[..., None], axis=1).view(-1, vocab_size)

	# use the ground-truth tokens as labels
	masked_tokens = np.take_along_axis(token_batch, mask_idxs, axis=1)
	masked_tokens = masked_tokens.view(-1).to(model.device)

	loss = loss_scale * F.cross_entropy(masked_logits, masked_tokens)
	loss.backward()
	optimizer.step()

	return loss, masked_logits, masked_tokens


def mlm_train_epoch(model, optimizer, train_loader, mask_ratio):
	metrics = dict(
		train_loss=0.,
		train_perplexity=0.,
	)
	model.train()
	for minibatch in train_loader:
		if isinstance(minibatch, tuple):
			token_batch = minibatch[0]
		else:
			assert torch.is_tensor(minibatch)
			token_batch = minibatch

		loss, masked_logits, masked_tokens = mlm_train_step(model, optimizer, token_batch, mask_ratio)

		# logging
		log_prob = F.log_softmax(masked_logits, dim=-1)
		log_prob = np.take_along_axis(log_prob, masked_tokens.cpu().numpy()[..., None], axis=1)
		metrics['train_perplexity'] += 2 ** (
			-(log_prob / math.log(2)).mean().detach()
		) / len(train_loader)
		metrics['train_loss'] += loss.detach() / len(train_loader)
	metrics = {key: val.item() for key, val in metrics.items()}
	return metrics


def mlm_eval_epoch(model, eval_loader, mask_ratio, split):
	metrics = dict(
		perplexity=0.,
	)
	model.eval()
	for minibatch in eval_loader:
		if isinstance(minibatch, tuple):
			token_batch = minibatch[0]
		else:
			assert torch.is_tensor(minibatch)
			token_batch = minibatch

		# replace random tokens with mask token
		mask_idxs = sample_mask(token_batch, model.tokenizer, mask_ratio)
		masked_token_batch = token_batch.clone().to(model.device)
		np.put_along_axis(masked_token_batch, mask_idxs, model.tokenizer.masking_idx, axis=1)

		# get predicted logits for masked tokens
		logits, _ = model.logits_from_tokens(masked_token_batch)
		vocab_size = logits.shape[-1]
		masked_logits = np.take_along_axis(logits, mask_idxs[..., None], axis=1).view(-1, vocab_size)

		# use the ground-truth tokens as labels
		masked_tokens = np.take_along_axis(token_batch, mask_idxs, axis=1)
		masked_tokens = masked_tokens.view(-1).to(model.device)

		# logging
		log_prob = F.log_softmax(masked_logits, dim=-1)
		log_prob = np.take_along_axis(log_prob, masked_tokens.cpu().numpy()[..., None], axis=1)
		metrics['perplexity'] += 2 ** (
			-(log_prob / math.log(2)).mean().detach()
		) / len(eval_loader)

	metrics = {key: val.item() for key, val in metrics.items()}
	metrics = {f'{split}_{key}': val for key, val in metrics.items()}

	return metrics


def fit_masked_language_model(model, train_seqs, num_epochs, batch_size, lr, patience, mask_ratio, max_shift,
							  weights=None, log_prefix=''):

	# random translation data augmentation, apply tokenizer
	train_transform = []
	if max_shift > 0:
		train_transform.append(gfp_transforms.SequenceTranslation(max_shift))
	train_transform.append(gfp_transforms.StringToLongTensor(model.tokenizer))
	train_transform = torchvision.transforms.Compose(train_transform)

	# make dataset, dataloader
	train_dataset = gfp_dataset.TransformTensorDataset([train_seqs], train_transform)

	if weights is None:
		loader_kwargs = dict(batch_size=batch_size, shuffle=True)
	else:
		sampler = torch.utils.data.WeightedRandomSampler(weights, batch_size, replacement=True)
		batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
		loader_kwargs = dict(batch_sampler=batch_sampler)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, collate_fn=gfp_transforms.padding_collate_fn, **loader_kwargs
	)

	optimizer = torch.optim.Adam(model.param_groups(lr))
	lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, patience=math.ceil(patience / 2)
	)

	records = []
	best_score, best_epoch, best_weights = None, 0, None
	model.requires_grad_(True)
	for epoch in range(num_epochs):
		metrics = {}
		metrics.update(
			mlm_train_epoch(model, optimizer, train_loader, mask_ratio)
		)
		# use avg. train loss as convergence crit.
		lr_sched.step(metrics['train_loss'])
		best_score, best_epoch, best_weights, stop = check_early_stopping(
			model,
			best_score,
			best_epoch,
			best_weights,
			metrics['train_loss'],
			epoch + 1,
			patience,
			save_weights=True,
			)

		# logging
		metrics.update(dict(best_score=best_score, best_epoch=best_epoch))
		if len(log_prefix) > 0:
			metrics = {'/'.join((log_prefix, key)): val for key, val in metrics.items()}
		try:
			wandb.log(metrics)
		except:
			pass
		records.append(metrics)

		if stop:
			break

	model.load_state_dict(best_weights)
	model.requires_grad_(False)

	return records
