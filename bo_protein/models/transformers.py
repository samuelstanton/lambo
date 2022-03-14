import numpy as np
import torch
import torch.nn as nn
import esm
import math
import wandb
import torchvision

from torch.nn import functional as F

from collections import namedtuple

from tape import TAPETokenizer, ProteinBertForMaskedLM

from bo_protein.models.utils import sample_mask
from bo_protein.utils import batched_call, AMINO_ACIDS
from bo_protein.gfp_data import transforms as gfp_transforms
from bo_protein.gfp_data import dataset as gfp_dataset
from bo_protein.models.trainer import check_early_stopping
from bo_protein.models.lanmt import FunctionHead


ESMObject = namedtuple('ESM_OBJECT', ['model', 'alphabet', 'converter', 'batch_size'])
BERTObject = namedtuple(
	'BERTObject',
	[
		'model',
		'tokenizer',
		'batch_size',
		'embedding_size',
		'padding_idx',
	]
)


def init_esm(batch_size=None):
	model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
	converter = alphabet.get_batch_converter()
	if torch.cuda.is_available():
		model = model.cuda()
	return ESMObject(model, alphabet, converter, batch_size)


def init_bert(batch_size=None):
	model = ProteinBertForMaskedLM.from_pretrained('bert-base')
	tokenizer = TAPETokenizer("iupac")
	return BERTObject(model, tokenizer, batch_size, 768, 0)


class MLMWrapper(nn.Module):
	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

	def __init__(self, batch_size, num_epochs, patience, lr, mask_ratio, max_shift, **kwargs):
		super().__init__()
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.patience = patience
		self.lr = lr
		self.mask_ratio = mask_ratio
		self.max_shift = max_shift

	def forward(self, inputs):
		if isinstance(inputs, np.ndarray):
			tokens = self.str_to_tokens(inputs)
		else:
			tokens = inputs
		token_features = self.get_token_features(tokens)
		return self.pool_features(tokens, token_features)

	def pool_features(self, tokens, token_features, ignore_idxs=None):
		# ignore_idxs = [
		# 	self.tokenizer.padding_idx,
		# 	self.tokenizer.eos_idx
		# ] if ignore_idxs is None else ignore_idxs
		src_mask = tokens.ne(self.tokenizer.padding_idx).float()
		pooling_mask = src_mask * tokens.ne(self.tokenizer.eos_idx).float()
		_, pooled_features = self.function_head(token_features, src_mask, pooling_mask)
		# src_tok_features, padding_mask = src_mask, pooling_mask = src_mask
		# pooled_features = pool_features(tokens, token_features, ignore_idxs=ignore_idxs)
		return pooled_features

	def fit(self, train_seqs, weights=None, num_epochs=None, log_prefix=''):
		num_epochs = self.num_epochs if num_epochs is None else num_epochs
		records = fit_masked_language_model(
			model=self,
			train_seqs=train_seqs,
			num_epochs=num_epochs,
			batch_size=self.batch_size,
			lr=self.lr,
			patience=self.patience,
			mask_ratio=self.mask_ratio,
			max_shift=self.max_shift,
			weights=weights,
			log_prefix=log_prefix,
		)
		return records


class BERTTokenizer(TAPETokenizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.special_vocab = list(set(self.vocab) - set(AMINO_ACIDS))
		self.non_special_vocab = list(set(self.vocab) - set(self.special_vocab))
		self.masking_idx = self.convert_token_to_id("<mask>")
		self.padding_idx = self.convert_token_to_id("<pad>")
		self.bos_idx = self.convert_token_to_id("<cls>")
		self.eos_idx = self.convert_token_to_id("<sep>")

	def encode(self, text: str):
		return super().encode(text)

	def decode(self, token_ids):
		tokens = []
		for t_id in token_ids:
			token = self.convert_id_to_token(t_id)
			if token in self.special_vocab and token not in ["<mask>", "<unk>"]:
				continue
			tokens.append(token)
		return ''.join(tokens)


# TODO finish refactor
class BERTWrapper(MLMWrapper):
	def __init__(self, out_dim, batch_size, finetune=False, **kwargs):
		super().__init__(batch_size, **kwargs)

		self.model = ProteinBertForMaskedLM.from_pretrained('bert-base')
		self.tokenizer = BERTTokenizer("iupac")
		self.embedding_size = 768
		self.function_head = FunctionHead(
			self.embedding_size, out_dim, None, None, dropout_p=0.1, num_heads=12, type='mha'
		)
		# self.linear = nn.Linear(self.embedding_size, out_dim)
		self.finetune = finetune

	def get_token_idx(self, token):
		return self.tokenizer.convert_token_to_id(token)

	def get_token(self, idx):
		return self.tokenizer.convert_id_to_token(idx)

	def get_token_features(self, tokens):
		if self.finetune:
			result_batches = batched_call(self.model.bert, tokens, self.batch_size)
		else:
			with torch.no_grad():
				result_batches = batched_call(self.model.bert, tokens, self.batch_size)

		token_features = torch.cat([batch[0] for batch in result_batches])
		return token_features

	def logits_from_tokens(self, tokens):
		token_features = self.get_token_features(tokens)
		return self.logits_from_features(token_features)

	def logits_from_features(self, token_features):
		result_batches = batched_call(self.model.mlm, token_features, self.batch_size, targets=None)
		logits = torch.cat([batch[0] for batch in result_batches])
		return logits

	def param_groups(self, lr, weight_decay=0.):
		groups = [
			dict(params=self.function_head.parameters(), lr=lr, weight_decay=weight_decay),
			dict(params=self.model.parameters(), lr=1e-4, betas=(0., 1e-2))
		]
		return groups

	def fit(self, train_seqs, weights=None, log_prefix=None):
		log_prefix = 'tape_bert' if log_prefix is None else log_prefix
		return super().fit(train_seqs, weights, log_prefix)


# TODO finish refactor
class ESMWrapper(MLMWrapper):
	def __init__(self, out_dim, batch_size, finetune=False, **kwargs):
		super().__init__(batch_size, **kwargs)

		self.model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
		self.alphabet = alphabet
		self.tokenizer = ESMTokenizer(alphabet)
		self.batch_converter = alphabet.get_batch_converter()

		self.embedding_size = 1280
		self.function_head = FunctionHead(
			self.embedding_size, out_dim, None, None, dropout_p=0.1, num_heads=12, type='mha'
		)
		# self.linear = nn.Linear(self.embedding_size, out_dim)
		self.finetune = finetune

	# def tokens_to_str(self, token_id_array):
	# 	str_array = np.array([
	# 		self.tokenizer.decode(token_ids) for token_ids in token_id_array
	# 	])
	# 	return str_array
	#
	# def str_to_tokens(self, str_array):
	# 	_, _, tokens = self.batch_converter([(f"protein{i}", s) for i, s in enumerate(str_array)])
	# 	return tokens

	def get_token_idx(self, token):
		return self.alphabet.get_idx(token)

	def get_token(self, idx):
		return self.alphabet.get_tok(idx)

	def get_token_features(self, tokens):
		if self.finetune:
			result_batches = batched_call(self.model, tokens, self.batch_size, repr_layers=[33])
		else:
			with torch.no_grad():
				result_batches = batched_call(self.model, tokens, self.batch_size, repr_layers=[33])
		token_features = torch.cat([batch["representations"][33] for batch in result_batches])
		return token_features

	def logits_from_tokens(self, tokens):
		result_batches = batched_call(self.model, tokens, self.batch_size)
		logits = torch.cat([batch["logits"] for batch in result_batches])
		return logits

	def logits_from_features(self, token_features):
		result_batches = batched_call(self.model.lm_head, token_features, self.batch_size)
		logits = torch.cat(result_batches)
		return logits

	def param_groups(self, lr, weight_decay=0.):
		groups = [
			dict(params=self.function_head.parameters(), lr=lr, weight_decay=weight_decay),
			dict(params=self.model.parameters(), lr=1e-4, betas=(0., 1e-2))
		]
		return groups

	def fit(self, train_seqs, weights=None, log_prefix=None):
		log_prefix = 'esm_bert' if log_prefix is None else log_prefix
		return super().fit(train_seqs, weights, log_prefix)


class ESMTokenizer:
	def __init__(self, alphabet):
		self.alphabet = alphabet
		self.batch_converter = alphabet.get_batch_converter()
		self.masking_idx = alphabet.mask_idx
		self.padding_idx = alphabet.padding_idx
		self.bos_idx = alphabet.bos_idx
		self.eos_idx = alphabet.eos_idx
		self.special_vocab = list(set(alphabet.all_toks) - set(AMINO_ACIDS))
		self.non_special_vocab = list(set(alphabet.all_toks) - set(self.special_vocab))

	def convert_token_to_id(self, token):
		"""
		syntactic sugar for compatibility
		"""
		return self.alphabet.get_idx(token)

	def convert_id_to_token(self, id):
		return self.alphabet.get_tok(id)

	def encode(self, x):
		return self.batch_converter([(None, x)])[-1][0].cpu().numpy()

	def decode(self, token_ids):
		tokens = []
		for t_id in token_ids:
			token = self.alphabet.get_tok(t_id)
			if token in self.special_vocab and token not in ["<mask>", "<unk>"]:
				continue
			tokens.append(token)
		return ''.join(tokens)


def pool_features(tokens, token_features, ignore_idxs):
	mask = torch.ones_like(tokens).float()
	for idx in ignore_idxs:
		mask *= tokens.ne(idx)
	mask = mask.unsqueeze(-1).float()
	pooled_features = (mask * token_features).sum(-2) / (mask.sum(-2) + 1e-6)

	return pooled_features


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
