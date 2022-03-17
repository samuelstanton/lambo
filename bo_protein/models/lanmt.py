import torch
from torch.nn import functional as F
import numpy as np
import math
import torchvision
import wandb

from bo_protein.models.mlm import sample_mask
from bo_protein import dataset as gfp_dataset, transforms as gfp_transforms
from bo_protein.models.shared_elements import check_early_stopping


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
