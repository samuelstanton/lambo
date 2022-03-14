import numpy as np
import torch
from torch import nn as nn
import math

from bo_protein.models.transformers import MLMWrapper, pool_features
from bo_protein.models.nn_models import mCNN
from bo_protein.utils import RESIDUE_ALPHABET, batched_call, AMINO_ACIDS, ResidueTokenizer
from bo_protein.gfp_data import utils as gfp_utils
from bo_protein.gfp_data.transforms import padding_collate_fn
from bo_protein.models.lanmt import PositionalEncoding


class LinearEncoder(nn.Module):
    def __init__(self, out_dim, embed_dim, tokenizer, max_len, *args, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer.full_vocab),
            embedding_dim=embed_dim,
            padding_idx=tokenizer.padding_idx,
        )
        self.pos_encoder = PositionalEncoding(
            embed_dim, dropout=0., max_len=max_len, batch_first=True
        )
        self.proj_head = nn.Linear(embed_dim, out_dim)
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.max_len = max_len

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            src_tok_idxs = self.tokenizer.encode(inputs)
        else:
            src_tok_idxs = inputs
        return self.forward(src_tok_idxs)

    def forward(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]

        src_mask = src_tok_idxs.ne(self.tokenizer.padding_idx).float()
        src_tok_features = self.embedding(src_tok_idxs)
        src_tok_features = self.pos_encoder(src_tok_features * math.sqrt(self.embed_dim))
        pooling_mask = src_mask * src_tok_idxs.ne(self.tokenizer.eos_idx).float()
        pooling_mask = pooling_mask.unsqueeze(-1)
        pooled_features = (pooling_mask * src_tok_features).sum(-2) / (pooling_mask.sum(-2) + 1e-6)
        return self.proj_head(pooled_features)

    def param_groups(self, lr, weight_decay=0.):
        return [dict(params=self.parameters(), lr=lr, weight_decay=weight_decay)]


class CNNEncoder(MLMWrapper):
    def __init__(self, out_dim, model_kwargs, tokenizer, batch_size, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.model = mCNN(
            tokenizer=tokenizer,
            out_dim=out_dim,
            **model_kwargs
        )
        self.embedding_size = self.model.embed_dim
        self.tokenizer = tokenizer
        self.logit_layer = nn.Linear(self.embedding_size, len(tokenizer.full_vocab))
        # self.logit_bias = nn.Parameter(data=torch.zeros(len(tokenizer.full_vocab)))
        self._mask = None

    @property
    def linear(self):
        return self.model.linear

    def mlm_head(self, token_features, mask):
        decoder_features = self.model.dec_tok_features(token_features, mask)
        reshaped_features = decoder_features.flatten(end_dim=-2)
        logits = self.logit_layer(reshaped_features)

        # if the input/output embeddings are tied, use this
        # reshaped_features = token_features.unsqueeze(-1).flatten(end_dim=-3)
        # weight = self.model.embedding.weight.expand(reshaped_features.size(0), -1, -1)
        # logits = weight.bmm(reshaped_features).squeeze(-1) + self.logit_bias

        logits = logits.view(*token_features.shape[:-1], -1)
        return logits

    def get_token_idx(self, token):
        return self.tokenizer.convert_token_to_id(token)

    def get_token(self, idx):
        return self.tokenizer.convert_id_to_token(idx)

    def get_token_features(self, tokens):
        token_features, self._mask = self.model.enc_tok_features(tokens)
        # result_batches = batched_call(self.model.encode_token_features, tokens, self.batch_size)  # --> (features, mask)
        # token_features = torch.cat([batch[0] for batch in result_batches])  # (b, n, c)
        # self._mask = torch.cat([batch[1] for batch in result_batches])
        return token_features

    def logits_from_tokens(self, tokens):
        token_features = self.get_token_features(tokens)
        return self.logits_from_features(token_features)

    def logits_from_features(self, token_features):
        # result_batches = batched_call(self.mlm_head, token_features, self.batch_size)  # --> logits
        # logits = torch.cat([batch for batch in result_batches])
        logits = self.mlm_head(token_features, self._mask)
        return logits

    def param_groups(self, lr, weight_decay=0.):
        groups = [
            dict(params=self.model.encoder.parameters(), lr=lr, weight_decay=weight_decay, betas=(0., 1e-2)),
            dict(params=self.model.decoder.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.logit_layer.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.linear.parameters(), lr=lr, weight_decay=weight_decay),
            dict(params=self.model.embedding.parameters(), lr=lr, weight_decay=weight_decay),
        ]
        return groups

    def fit(self, train_seqs, weights=None, num_epochs=None, log_prefix=None):
        log_prefix = 'cnn_mlm' if log_prefix is None else log_prefix
        return super().fit(train_seqs, weights, num_epochs, log_prefix)
