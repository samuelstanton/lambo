import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import math
import torchvision
import wandb

from bo_protein.models.utils import sample_mask
from bo_protein.gfp_data import transforms as gfp_transforms
from bo_protein.gfp_data import dataset as gfp_dataset
from bo_protein.models.trainer import check_early_stopping
from bo_protein.models.masked_layers import mResidualBlock


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
        :return:
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


def corrupt_tok_idxs(tgt_tok_idxs, tokenizer, max_len_delta, select_idxs=None):
    viable_idxs = np.array(tokenizer.non_special_idxs + [tokenizer.masking_idx])

    if select_idxs is None:
        # sample position indexes
        rand_idxs = sample_mask(tgt_tok_idxs, tokenizer, mask_size=max_len_delta)
        # sample length changes
        len_deltas = np.random.randint(
            0, max_len_delta + 1, (tgt_tok_idxs.size(0),)
        )
        # sample tokens for insertion/substitution
        rand_tok_idxs = torch.tensor(
            np.random.choice(viable_idxs, rand_idxs.shape, replace=True)
        )
    else:
        # sample tokens for insertion/substitution
        rand_tok_idxs = torch.tensor(
            np.random.choice(viable_idxs, select_idxs.shape, replace=True)
        )

    src_tok_idxs = []
    for row_idx, tgt_row in enumerate(tgt_tok_idxs):
        src_row = tgt_row.clone().cpu()

        # select seq. positions to corrupt
        if select_idxs is None:
            delta = len_deltas[row_idx]
            if delta == 0:
                src_tok_idxs.append(src_row)
                continue
            selected = rand_idxs[row_idx, :delta]
            new_tok_idxs = rand_tok_idxs[row_idx, :delta]
        else:
            selected = select_idxs[row_idx]
            new_tok_idxs = rand_tok_idxs[row_idx]

        p = np.random.rand()
        if p < 0.33:
            # print('deletion')
            src_tok_idxs.append(np.delete(src_row, selected, axis=0))
        elif p < 0.66:
            # print('insertion')
            src_tok_idxs.append(np.insert(src_row, selected, new_tok_idxs, axis=0))
        else:
            # print('substitution')
            np.put_along_axis(src_row, selected, new_tok_idxs, axis=0)
            src_tok_idxs.append(src_row)

    # pad sequences back to same length
    max_len = max([src_row.size(0) for src_row in src_tok_idxs])
    for row_idx, src_row in enumerate(src_tok_idxs):
        if src_row.size(0) == max_len:
            continue
        padding = torch.tensor(
            [tokenizer.padding_idx] * (max_len - src_row.size(0))
        ).to(src_row)
        src_tok_idxs[row_idx] = torch.cat([src_row, padding])

    src_tok_idxs = torch.stack(src_tok_idxs).to(tgt_tok_idxs)

    return src_tok_idxs


def lanmt_train_step(model, optimizer, tgt_tok_idxs, loss_scale=1.):
    optimizer.zero_grad(set_to_none=True)

    # corrupt random tokens
    src_tok_idxs = corrupt_tok_idxs(tgt_tok_idxs, model.tokenizer, model.max_len_delta)
    src_tok_idxs = src_tok_idxs.to(model.device)
    tgt_tok_idxs = tgt_tok_idxs.to(model.device)

    # get features for corrupted seqs
    src_tok_features, src_mask = model.enc_tok_features(src_tok_idxs)
    src_lens = src_mask.float().sum(-1)
    # get features for target seqs
    tgt_lens = tgt_tok_idxs.ne(model.tokenizer.padding_idx).float().sum(-1)
    _, tgt_tok_features, tgt_mask, len_delta_logits = model.dec_tok_features(
        src_tok_features, src_mask, lat_tok_features=None, tgt_lens=tgt_lens
    )
    tgt_tok_logits = model.tgt_tok_logits(tgt_tok_features)

    tok_loss = F.cross_entropy(
        tgt_tok_logits.flatten(end_dim=-2), tgt_tok_idxs.flatten(), ignore_index=model.tokenizer.padding_idx
    )
    len_deltas = (tgt_lens - src_lens).long()
    len_targets = len_deltas + model.max_len_delta
    len_loss = F.cross_entropy(len_delta_logits, len_targets)

    loss = loss_scale * (tok_loss + len_loss)
    loss.backward()
    optimizer.step()

    return loss, tgt_tok_logits, tgt_tok_idxs


def lanmt_train_epoch(model, optimizer, train_loader):
    metrics = dict(
        train_loss=0.,
        train_perplexity=0.,
    )
    model.train()
    for minibatch in train_loader:
        if isinstance(minibatch, tuple):
            tgt_tok_idxs = minibatch[0]
        else:
            assert torch.is_tensor(minibatch)
            tgt_tok_idxs = minibatch

        loss, tgt_tok_logits, tgt_tok_idxs = lanmt_train_step(model, optimizer, tgt_tok_idxs)

        # logging
        tgt_mask = tgt_tok_idxs.ne(model.tokenizer.padding_idx).float()
        log_prob = F.log_softmax(tgt_tok_logits, dim=-1)
        log_prob = np.take_along_axis(log_prob, tgt_tok_idxs.cpu().numpy()[..., None], axis=-1).squeeze(-1)
        log_prob *= tgt_mask

        metrics['train_perplexity'] += 2 ** (
            -(log_prob / math.log(2)).sum() / tgt_mask.sum()
        ).item() / len(train_loader)
        metrics['train_loss'] += loss.item() / len(train_loader)
    return metrics


def lanmt_eval_epoch(model, eval_loader, split):
    metrics = dict(
        perplexity=0.,
    )
    model.eval()
    for minibatch in eval_loader:
        if isinstance(minibatch, tuple):
            tgt_tok_idxs = minibatch[0]
        else:
            assert torch.is_tensor(minibatch)
            tgt_tok_idxs = minibatch

        # corrupt random tokens
        src_tok_idxs = corrupt_tok_idxs(tgt_tok_idxs, model.tokenizer, model.max_len_delta)
        src_tok_idxs = src_tok_idxs.to(model.device)
        tgt_tok_idxs = tgt_tok_idxs.to(model.device)

        # get features for corrupted seqs
        src_tok_features, src_mask = model.enc_tok_features(src_tok_idxs)
        # src_lens = src_mask.float().sum(-1)
        # get features for target seqs
        tgt_lens = tgt_tok_idxs.ne(model.tokenizer.padding_idx).float().sum(-1)
        _, tgt_tok_features, tgt_mask, len_delta_logits = model.dec_tok_features(
            src_tok_features, src_mask, lat_tok_features=None, tgt_lens=tgt_lens
        )
        tgt_tok_logits = model.tgt_tok_logits(tgt_tok_features)

        # logging
        # tgt_mask = tgt_tok_idxs.ne(model.tokenizer.padding_idx).float()
        log_prob = F.log_softmax(tgt_tok_logits, dim=-1)
        log_prob = np.take_along_axis(log_prob, tgt_tok_idxs.cpu().numpy()[..., None], axis=-1).squeeze(-1)
        log_prob *= tgt_mask

        metrics['perplexity'] += 2 ** (
            -(log_prob / math.log(2)).sum() / tgt_mask.sum()
        ).item() / len(eval_loader)

    metrics = {f'{split}_{key}': val for key, val in metrics.items()}

    return metrics


def fit_lanmt_model(model, train_seqs, num_epochs, batch_size, lr, patience, max_shift,
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
            lanmt_train_epoch(model, optimizer, train_loader)
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
        except Exception:
            pass
        records.append(metrics)

        if stop:
            break

    model.load_state_dict(best_weights)
    model.requires_grad_(False)

    return records


class LANMTWrapper(nn.Module):
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
        # src_mask = src_tok_idxs.ne(self.model.tokenizer.padding_idx)
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
        # result_batches = batched_call(self.model.encode_token_features, tokens, self.batch_size)  # --> (features, mask)
        # token_features = torch.cat([batch[0] for batch in result_batches])  # (b, n, c)
        # self._mask = torch.cat([batch[1] for batch in result_batches])
        return src_tok_features, src_mask

    def logits_from_tokens(self, src_tok_idxs):
        src_tok_features, src_mask = self.get_token_features(src_tok_idxs)
        tgt_tok_logits, tgt_mask = self.logits_from_features(src_tok_features, src_mask, lat_tok_features=None)
        return tgt_tok_logits, tgt_mask

    def logits_from_features(self, src_tok_features, src_mask, lat_tok_features, tgt_lens=None):
        # result_batches = batched_call(self.mlm_head, token_features, self.batch_size)  # --> logits
        # logits = torch.cat([batch for batch in result_batches])
        # src_mask = src_tok_idxs.ne(self.tokenizer.padding_idx)
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
