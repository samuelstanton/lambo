import math

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from bo_protein.models.lanmt import fit_lanmt_model
from bo_protein.models.masked_layers import mResidualBlock


class LanguageModel(nn.Module):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def __init__(self, model, batch_size, num_epochs, patience, lr, max_shift, mask_ratio, **kwargs):
        super().__init__()
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr = lr
        self.max_shift = max_shift
        self.tokenizer = model.tokenizer
        self.mask_ratio = mask_ratio

    def forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            tok_idxs = self.str_to_tokens(inputs)
        else:
            tok_idxs = inputs
        return self.model(tok_idxs)

    def pool_features(self, src_tok_features, src_mask):
        lat_tok_features, pooled_features = self.model.function_head(
            src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
        )
        return lat_tok_features, pooled_features

    def fit(self, train_seqs, weights=None, num_epochs=None, log_prefix='lanmt'):
        num_epochs = self.num_epochs if num_epochs is None else num_epochs
        records = fit_lanmt_model(
            model=self.model,
            train_seqs=train_seqs,
            num_epochs=num_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            max_shift=self.max_shift,
            weights=weights,
            log_prefix=log_prefix,
        )
        return records

    def get_token_idx(self, token):
        return self.model.tokenizer.convert_token_to_id(token)

    def get_token(self, idx):
        return self.model.tokenizer.convert_id_to_token(idx)

    def get_token_features(self, src_tok_idxs):
        src_tok_features, src_mask = self.model.enc_tok_features(src_tok_idxs)
        return src_tok_features, src_mask

    def logits_from_tokens(self, src_tok_idxs):
        src_tok_features, src_mask = self.get_token_features(src_tok_idxs)
        tgt_tok_logits, tgt_mask = self.logits_from_features(src_tok_features, src_mask, lat_tok_features=None)
        return tgt_tok_logits, tgt_mask

    def logits_from_features(self, src_tok_features, src_mask, lat_tok_features, tgt_lens=None):
        lat_tok_features, tgt_tok_features, tgt_mask, _ = self.model.dec_tok_features(
            src_tok_features, src_mask, lat_tok_features, tgt_lens
        )
        tgt_tok_logits = self.model.tgt_tok_logits(tgt_tok_features)
        return tgt_tok_logits, tgt_mask

    def sample_tgt_tok_idxs(self, tgt_tok_logits, tgt_mask, temp=1.):
        batch_size, num_tokens = tgt_mask.shape
        tgt_lens = tgt_mask.float().sum(-1).long()
        tgt_tok_logits /= temp

        # don't sample special tokens
        non_viable_idxs = np.array(self.tokenizer.special_idxs)[None, None, :]
        np.put_along_axis(tgt_tok_logits, non_viable_idxs, -1e10, axis=-1)

        tgt_tok_idxs = torch.full(tgt_mask.size(), self.tokenizer.padding_idx)
        tgt_tok_idxs = tgt_tok_idxs.to(tgt_mask).long()
        tok_dist = torch.distributions.Categorical(logits=tgt_tok_logits)
        sample_tok_idxs = tok_dist.sample()

        tgt_tok_idxs += tgt_mask * sample_tok_idxs

        tgt_tok_idxs[:, 0] = self.tokenizer.bos_idx
        tgt_tok_idxs[torch.arange(batch_size), tgt_lens - 1] = self.tokenizer.eos_idx

        logit_entropy = -(
            F.softmax(tgt_tok_logits, dim=-1) * F.log_softmax(tgt_tok_logits, dim=-1)
        ).sum(-1)
        logit_entropy *= tgt_mask.float()
        logit_entropy = logit_entropy.sum() / tgt_mask.float().sum()

        return tgt_tok_idxs, logit_entropy

    def param_groups(self, *args, **kwargs):
        return self.model.param_groups(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len + 1, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.transpose(1, 0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class FunctionHead(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, layernorm, dropout_p, num_heads, type):
        super().__init__()
        if type == 'conv':
            self.att_layer = mResidualBlock(input_dim, input_dim, kernel_size, layernorm, dropout_p)
        elif type == 'mha':
            self.att_layer = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout_p, batch_first=True)
            self.dropout = nn.Dropout(dropout_p)
        else:
            raise ValueError
        self.type = type
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
            self,
            tok_features: Tensor,
            padding_mask: Tensor,
            pooling_mask: Tensor,
    ):
        """
        :param tok_features: (batch_size, num_tokens, input_dim)
        :param padding_mask: (batch_size, num_tokens) True to NOT ignore
        :param pooling_mask: (batch_size, num_tokens) True to NOT ignore
        """
        # conv layers expect inputs with shape (batch_size, input_dim, num_tokens)
        if self.type == 'conv':
            tok_features, _ = self.att_layer((tok_features.permute(0, 2, 1), padding_mask))
            tok_features = tok_features.permute(0, 2, 1)
        else:
            key_padding_mask = ~padding_mask.bool()  # invert mask
            tok_features, _ = self.att_layer(
                tok_features, tok_features, tok_features, key_padding_mask, need_weights=False
            )
            tok_features = self.dropout(F.gelu(tok_features))

        pooling_mask = pooling_mask.unsqueeze(-1).float()
        pooled_features = (pooling_mask * tok_features).sum(-2) / (pooling_mask.sum(-2) + 1e-6)
        pooled_features = self.linear(pooled_features)
        return tok_features, pooled_features


class LengthHead(nn.Module):
    def __init__(self, input_dim, max_len_delta):
        super().__init__()
        num_classes = max_len_delta * 2 + 1
        self.linear = nn.Linear(input_dim, num_classes)
        self.max_len_delta = max_len_delta

    def forward(self, tok_features, pooling_mask):
        pooling_mask = pooling_mask.unsqueeze(-1).float()
        pooled_features = (pooling_mask * tok_features).sum(-2) / (pooling_mask.sum(-2) + 1e-6)
        logits = self.linear(pooled_features)
        return logits

    def sample(self, src_lens, logits):
        if self.max_len_delta == 0:
            return src_lens
        tgt_len_dist = torch.distributions.Categorical(logits=logits)
        len_deltas = tgt_len_dist.sample()
        len_deltas -= self.max_len_delta
        return src_lens + len_deltas


class LengthTransform(nn.Module):
    """
    monotonic location-based attention mechanism from
    https://arxiv.org/abs/1908.07181
    """
    def __init__(self):
        super().__init__()
        self.register_parameter('lengthscale', nn.Parameter(torch.tensor(1.)))

    def forward(
            self,
            src_tok_features: Tensor,
            src_mask: Tensor,
            tgt_lens: Tensor,
    ):
        """
        :param src_tok_features: (batch_size, num_src_tokens, embed_dim)
        :param src_mask: (batch_size, num_src_tokens)
        :param tgt_lens: (batch_size,)
        :return:
        """
        batch_size, num_src_tokens = src_mask.shape
        src_lens = src_mask.float().sum(-1)
        tgt_lens = tgt_lens.to(src_lens)

        if torch.all(src_lens == tgt_lens):
            return src_tok_features, src_mask.bool()

        src_arange = torch.arange(num_src_tokens).to(src_mask.device)
        src_arange = src_arange.expand(batch_size, -1).unsqueeze(-1).float()  # (batch_size, num_src_tokens, 1)

        tgt_arange = torch.arange(tgt_lens.max()).to(src_mask.device)
        tgt_arange = tgt_arange.expand(batch_size, -1).unsqueeze(-2).float()  # (batch_size, 1, num_tgt_tokens)

        len_ratio = src_lens / tgt_lens  # (batch_size,)
        len_ratio = len_ratio.view(-1, 1, 1)

        sq_diff = (src_arange - len_ratio * tgt_arange) ** 2
        sq_diff = sq_diff.to(self.lengthscale)

        logits = -sq_diff / (2 * self.lengthscale ** 2)  # (batch_size, num_src_tokens, num_tgt_tokens)
        logits = src_mask[..., None] * logits - (1 - src_mask[..., None]) * 1e10
        weights = F.softmax(logits, dim=-2)
        weights = weights.unsqueeze(-1).to(src_tok_features)

        tgt_token_features = src_tok_features.unsqueeze(-2)
        tgt_token_features = weights * tgt_token_features  # (batch_size, num_src_tokens, num_tgt_tokens, embed_dim)
        tgt_token_features = tgt_token_features.sum(-3)

        tgt_mask = (tgt_arange.squeeze(-2) < tgt_lens.unsqueeze(-1))

        return tgt_token_features, tgt_mask
