import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import logging

from .masked_layers import Apply, mMaxPool1d, mAvgPool1d, mConvNormAct, LayerNorm1d, mResidualBlock
from .transformers import pool_features
from .lanmt import LengthTransform, FunctionHead, LengthHead, PositionalEncoding


def swish(x):
    return x * x.sigmoid()


class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def ConvBNrelu(in_channels, out_channels, layernorm=False, ksize=5, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, ksize, padding=ksize // 2, stride=stride),
        LayerNorm1d(out_channels) if layernorm else nn.BatchNorm1d(out_channels),
        Expression(swish),  # nn.ReLU()
    )


import inspect


def local_kwargs(kwargs, f):
    """Return the kwargs from dict that are inputs to function f."""
    s = inspect.signature(f)
    p = s.parameters
    if next(reversed(p.values())).kind == inspect.Parameter.VAR_KEYWORD:
        return kwargs
    if len(kwargs) < len(p):
        return {k: v for k, v in kwargs.items() if k in p}
    return {k: kwargs[k] for k in p.keys() if k in kwargs}


class mySequential(nn.Sequential):
    def forward(self, x, **kwargs):
        for module in self._modules.values():
            x = module(x, **local_kwargs(kwargs, module.forward))
        return x


class CNN(nn.Module):
    def __init__(self, dict_size=20, k=128, p=0.5, out_dim=1, layernorm=False, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Embedding(dict_size, k),
            Expression(lambda x: x.permute(0, 2, 1)),  # (B,N,C) -> (B,C,N)
            ConvBNrelu(k, k, layernorm),
            ConvBNrelu(k, 2 * k, layernorm),
            nn.MaxPool1d(2),
            nn.Dropout2d(p),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            nn.MaxPool1d(2),
            nn.Dropout2d(p),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            nn.Dropout2d(p),
            Expression(lambda u: u.mean(-1)),
            nn.Linear(2 * k, out_dim),
        )

    def forward(self, x):
        if self.out_dim == 1:
            return self.net(x)[..., 0]
        else:
            return self.net(x)


class CNNbinned(nn.Module):
    def __init__(
        self,
        bin_avgs,
        dict_size=20,
        k=128,
        p=0.5,
        layernorm=False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(dict_size, k),
            Expression(lambda x: x.permute(0, 2, 1)),  # (B,N,C) -> (B,C,N)
            ConvBNrelu(k, k, layernorm),
            ConvBNrelu(k, 2 * k, layernorm),
            nn.MaxPool1d(2),
            nn.Dropout2d(p),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            nn.MaxPool1d(2),
            nn.Dropout2d(p),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            nn.Dropout2d(p),
            Expression(lambda u: u.mean(-1)),
            nn.Linear(2 * k, len(bin_avgs)),
        )
        self.bin_avgs = bin_avgs

    def forward(self, x, regress=False):
        """logits for classification into bins
        or Regression output by weighting bin avgs by probabilities"""
        # if regress: return self.bin_avgs.to(x.device)[self(x).max(-1)[1]]
        if regress:
            return (F.softmax(self(x), dim=-1) * self.bin_avgs.to(x.device)).sum(-1)
        return self.net(x)


class bCNN(nn.Module):
    def __init__(self, dict_size=20, k=128, p=0, layernorm=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(dict_size, k),
            Expression(lambda x: x.permute(0, 2, 1)),  # (B,N,C) -> (B,C,N)
            ConvBNrelu(k, k, layernorm),
            ConvBNrelu(k, 2 * k, layernorm),
            nn.MaxPool1d(2),
            nn.Dropout2d(p),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            nn.MaxPool1d(2),
            nn.Dropout2d(p),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            ConvBNrelu(2 * k, 2 * k, layernorm),
            nn.Dropout2d(p),
            Expression(lambda u: u.mean(-1)),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(2 * k, 2 * k), Expression(swish), nn.Linear(2 * k, k)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(2 * k, 2 * k), Expression(swish), nn.Linear(2 * k, 1)
        )

    def forward(self, x, proj=False):
        if proj:
            return self.projection_head(self.net(x))
        return self.prediction_head(self.net(x))[..., 0]


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
        # import pdb; pdb.set_trace()

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


class RNN(nn.Module):
    def __init__(self, dict_size=20, k=256, nlayer=2, bi=True, **kwargs):
        super().__init__()
        self.nlayer = nlayer
        self.gru = nn.GRU(k, k, nlayer, bidirectional=bi)
        self.embedding = nn.Embedding(dict_size, k)
        self.linear = nn.Linear(k * (1 + bi), 1)
        self.bi = bi

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        n, bs, k = x.shape
        h0 = torch.zeros(self.nlayer * (1 + self.bi), bs, k).to(x.device)
        out, hf = self.gru(x, h0)
        return self.linear(out.mean(0)).reshape(-1)
        # return self.linear(hf.permute(1,0,2).reshape(bs,-1)).reshape(-1)


class MLP(nn.Module):
    def __init__(self, chin=128, k=128, nlayers=3):
        super().__init__()
        chs = [chin] + nlayers * [k]
        self.net = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(cin, cout), nn.ReLU())
                for cin, cout in zip(chs, chs[1:])
            ],
            nn.Linear(chs[-1], 1)
        )

    def forward(self, x):
        y = self.net(x)
        return y[..., 0]


############################################################################
# BELOW LIFTED FROM https://github.com/locuslab/TCN/blob/master/TCN/tcn.py #
############################################################################


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, 
                       kernel_size=5, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(input_size, output_size)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y = self.tcn(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous().squeeze(-1)


class DecoderTCN(nn.Module):
    def __init__(self, input_size, output_size, latent_size, num_channels, 
                       kernel_size=5, dropout=0.2, emb_dropout=0.2):
        super(DecoderTCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size + latent_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(input_size, output_size)
#         self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, z):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        N, L_in = x.shape
        emb = self.drop(self.encoder(x))
#         print(z.mean(0))
        z_rep = z[:,None,:].expand(-1, L_in, -1)
        y = self.tcn(torch.cat([emb, z_rep], dim=2).transpose(1, 2))
#         print(y.shape)
#         print(y.mean((0,-1)))
#         print("\n")
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous().squeeze(-1)
