import copy
import math
import typing

import torch
from torch import nn as nn

from lambo.models.lm_elements import PositionalEncoding, FunctionHead, LengthHead, LengthTransform
from lambo.models.masked_layers import Apply, mResidualBlock


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_fn(name: str) -> typing.Callable:
    if name == "gelu":
        return gelu
    elif name == "relu":
        return torch.nn.functional.relu
    elif name == "swish":
        return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")


class LayerNorm(nn.Module):  # type: ignore
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def pool_features(tokens, token_features, ignore_idxs):
	mask = torch.ones_like(tokens, dtype=torch.float)
	for idx in ignore_idxs:
		mask *= tokens.ne(idx)
	mask = mask.unsqueeze(-1).to(token_features)
	pooled_features = (mask * token_features).sum(-2) / (mask.sum(-2) + 1e-6)

	return pooled_features


class mCNN(nn.Module):
    """1d CNN for sequences like CNN, but includes an additional mask
    argument (bs,n) that specifies which elements in the sequence are
    merely padded values."""

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def __init__(self, tokenizer, max_len, embed_dim=128, kernel_size=5, p=0.1, out_dim=1,
                 layernorm=False, latent_dim=8, max_len_delta=2, num_heads=2, **kwargs):
        super().__init__()
        vocab_size = len(tokenizer.full_vocab)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, p, max_len, batch_first=True)
        self.encoder = nn.Sequential(
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,N,C) -> (B,C,N)
            # mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm),
            mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
            mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
            mResidualBlock(embed_dim, latent_dim, kernel_size, layernorm, p),
            # Apply(nn.Dropout(p=p)),
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,C,N) -> (B,N,C)
        )
        self.decoder = nn.Sequential(
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,N,C) -> (B,C,N)
            mResidualBlock(2 * latent_dim, embed_dim, kernel_size, layernorm, p),
            mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
            mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
            # mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm),
            # Apply(nn.Dropout(p=p)),
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,C,N) -> (B,N,C)
        )

        self.length_transform = LengthTransform()
        self.function_head = FunctionHead(latent_dim, out_dim, kernel_size, layernorm, p, None, type='conv')
        self.length_head = LengthHead(latent_dim, max_len_delta)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.tokenizer = tokenizer
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.max_len_delta = max_len_delta

    def enc_tok_features(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]

        src_tok_features = self.embedding(src_tok_idxs) * math.sqrt(self.embed_dim)
        src_tok_features = self.pos_encoder(src_tok_features)
        src_mask = (src_tok_idxs != self.tokenizer.padding_idx).float()
        src_tok_features, _ = self.encoder((src_tok_features, src_mask))
        return src_tok_features, src_mask

    def dec_tok_features(self, src_tok_features, src_mask, lat_tok_features=None, tgt_lens=None):
        # internal features from function head
        if lat_tok_features is None:
            lat_tok_features, _ = self.function_head(
                src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
            )

        len_delta_logits = self.length_head(src_tok_features, src_mask)
        # predict target seq length if unknown
        if tgt_lens is None:
            src_lens = src_mask.float().sum(-1)
            tgt_lens = self.length_head.sample(src_lens, len_delta_logits)

        tgt_tok_features, tgt_mask = self.length_transform(
            src_tok_features=torch.cat([src_tok_features, lat_tok_features], dim=-1),
            src_mask=src_mask,
            tgt_lens=tgt_lens
        )
        tgt_tok_features, _ = self.decoder((tgt_tok_features, tgt_mask))

        return lat_tok_features, tgt_tok_features, tgt_mask, len_delta_logits

    def tgt_tok_logits(self, tgt_tok_features):
        reshaped_features = tgt_tok_features.flatten(end_dim=-2)
        logits = self.lm_head(reshaped_features)
        logits = logits.view(*tgt_tok_features.shape[:-1], -1)
        return logits

    def forward(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]
        src_tok_features, src_mask = self.enc_tok_features(src_tok_idxs)
        pooling_mask = src_mask * src_tok_idxs.ne(self.tokenizer.eos_idx)
        _, pooled_features = self.function_head(src_tok_features, src_mask, pooling_mask)
        return pooled_features

    def param_groups(self, lr, weight_decay=0.):
        shared_group = dict(params=[], lr=lr, weight_decay=weight_decay, betas=(0., 1e-2))
        other_group = dict(params=[], lr=lr, weight_decay=weight_decay)

        shared_names = ['embedding', 'pos_encoder', 'encoder', 'function_head']
        for p_name, param in self.named_parameters():
            prefix = p_name.split('.')[0]
            if prefix in shared_names:
                shared_group['params'].append(param)
            else:
                other_group['params'].append(param)

        return shared_group, other_group


class Transformer(nn.Module):
    """1d CNN for sequences like CNN, but includes an additional mask
    argument (bs,n) that specifies which elements in the sequence are
    merely padded values."""

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def __init__(self, tokenizer, max_len, embed_dim=64, ff_dim=256, num_heads=2, num_layers=4, p=0.1, out_dim=1,
                 latent_dim=16, max_len_delta=2, **kwargs):
        super().__init__()
        vocab_size = len(tokenizer.full_vocab)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, p, max_len, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=p, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=p, batch_first=True
            ),
            num_layers=num_layers,
        )

        self.length_transform = LengthTransform()
        self.function_head = FunctionHead(latent_dim, out_dim, None, None, p, num_heads, type='mha')
        self.length_head = LengthHead(latent_dim, max_len_delta)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.embed2latent = nn.Linear(embed_dim, latent_dim)
        self.latent2embed = nn.Linear(2 * latent_dim, embed_dim)

        self.tokenizer = tokenizer
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.max_len_delta = max_len_delta

    def enc_tok_features(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]

        src_tok_features = self.embedding(src_tok_idxs) * math.sqrt(self.embed_dim)
        src_tok_features = self.pos_encoder(src_tok_features)
        key_padding_mask = src_tok_idxs.eq(self.tokenizer.padding_idx)

        src_tok_features = self.encoder(src_tok_features, src_key_padding_mask=key_padding_mask)
        src_tok_features = self.embed2latent(src_tok_features)
        src_mask = (~key_padding_mask).float()

        return src_tok_features, src_mask

    def dec_tok_features(self, src_tok_features, src_mask, lat_tok_features=None, tgt_lens=None):
        # internal features from function head
        if lat_tok_features is None:
            lat_tok_features, _ = self.function_head(
                src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
            )

        len_delta_logits = self.length_head(src_tok_features, src_mask)
        # predict target seq length if unknown
        if tgt_lens is None:
            src_lens = src_mask.float().sum(-1)
            tgt_lens = self.length_head.sample(src_lens, len_delta_logits)

        tgt_tok_features, tgt_mask = self.length_transform(
            src_tok_features=torch.cat([src_tok_features, lat_tok_features], dim=-1),
            src_mask=src_mask,
            tgt_lens=tgt_lens
        )
        tgt_tok_features = self.latent2embed(tgt_tok_features)

        tgt_pad_mask = ~tgt_mask.bool()
        tgt_tok_features = self.decoder(
            tgt_tok_features,
            src_key_padding_mask=tgt_pad_mask,
        )

        return lat_tok_features, tgt_tok_features, tgt_mask, len_delta_logits

    def tgt_tok_logits(self, tgt_tok_features):
        reshaped_features = tgt_tok_features.flatten(end_dim=-2)
        logits = self.lm_head(reshaped_features)
        logits = logits.view(*tgt_tok_features.shape[:-1], -1)
        return logits

    def forward(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]
        src_tok_features, src_mask = self.enc_tok_features(src_tok_idxs)
        pooling_mask = src_mask * src_tok_idxs.ne(self.tokenizer.eos_idx)
        _, pooled_features = self.function_head(src_tok_features, src_mask, pooling_mask)
        return pooled_features

    def param_groups(self, lr, weight_decay=0.):
        shared_group = dict(params=[], lr=lr, weight_decay=weight_decay, betas=(0., 1e-2))
        other_group = dict(params=[], lr=lr, weight_decay=weight_decay)

        shared_names = ['embedding', 'pos_encoder', 'encoder', 'function_head']
        for p_name, param in self.named_parameters():
            prefix = p_name.split('.')[0]
            if prefix in shared_names:
                shared_group['params'].append(param)
            else:
                other_group['params'].append(param)

        return shared_group, other_group


class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def check_early_stopping(
        model,
        best_score,
        best_epoch,
        best_weights,
        curr_score,
        curr_epoch,
        patience,
        tol=1e-3,
        save_weights=True,
):
    eps = 1e-6
    stop = False
    if (
            best_score is None
            or (best_score - curr_score) / (abs(best_score) + eps) > tol
    ):
        best_score, best_epoch = curr_score, curr_epoch
    elif curr_epoch - best_epoch >= patience:
        stop = True
    else:
        pass

    if best_epoch == curr_epoch and save_weights:
        del best_weights
        model.cpu()  # avoid storing two copies of the weights on GPU
        best_weights = copy.deepcopy(model.state_dict())
        model.to(model.device)

    return best_score, best_epoch, best_weights, stop
