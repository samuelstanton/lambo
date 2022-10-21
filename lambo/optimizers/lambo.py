import hydra
import wandb
import pandas as pd
import time
import numpy as np
import torch
import random

from torch.nn import functional as F

from pymoo.factory import get_performance_indicator

from botorch.utils.multi_objective import infer_reference_point

from lambo.models.mlm import sample_tokens, evaluate_windows
from lambo.optimizers.pymoo import pareto_frontier, Normalizer
from lambo.models.shared_elements import check_early_stopping
from lambo.utils import weighted_resampling, DataSplit, update_splits, str_to_tokens, tokens_to_str, safe_np_cat
from lambo.models.lanmt import corrupt_tok_idxs


class LaMBO(object):
    def __init__(self, bb_task, tokenizer, encoder, surrogate, acquisition, num_rounds, num_gens,
                 lr, num_opt_steps, concentrate_pool, patience, mask_ratio, resampling_weight,
                 encoder_obj, optimize_latent, position_sampler, entropy_penalty,
                 window_size, latent_init, **kwargs):

        self.tokenizer = tokenizer
        self.num_rounds = num_rounds
        self.num_gens = num_gens
        self.concentrate_pool = concentrate_pool
        self._hv_ref = None
        self._ref_point = np.array([1] * bb_task.obj_dim)
        self.max_num_edits = bb_task.max_num_edits

        self.bb_task = hydra.utils.instantiate(bb_task, tokenizer=tokenizer, candidate_pool=[])

        self.encoder_config = encoder
        self.encoder = hydra.utils.instantiate(encoder, tokenizer=tokenizer)
        self.encoder_obj = encoder_obj

        self.surrogate_config = surrogate
        self.surrogate_model = hydra.utils.instantiate(surrogate, tokenizer=self.encoder.tokenizer,
                                                       encoder=self.encoder)
        self.acquisition = acquisition

        self.lr = lr
        self.num_opt_steps = num_opt_steps
        self.patience = patience
        self.mask_ratio = mask_ratio
        self.resampling_weight = resampling_weight
        self.optimize_latent = optimize_latent
        self.position_sampler = position_sampler
        self.entropy_penalty = entropy_penalty
        self.window_size = window_size
        self.latent_init = latent_init

        self.active_candidates = None
        self.active_targets = None
        self.train_split = DataSplit()
        self.val_split = DataSplit()
        self.test_split = DataSplit()

    def optimize(self, candidate_pool, pool_targets, all_seqs, all_targets, log_prefix=''):
        batch_size = self.bb_task.batch_size
        target_min = all_targets.min(axis=0).copy()
        target_range = all_targets.max(axis=0).copy() - target_min
        hypercube_transform = Normalizer(
            loc=target_min + 0.5 * target_range,
            scale=target_range / 2.,
        )
        new_seqs = all_seqs.copy()
        new_targets = all_targets.copy()

        is_feasible = self.bb_task.is_feasible(candidate_pool)
        pool_candidates = candidate_pool[is_feasible]
        pool_targets = pool_targets[is_feasible]
        pool_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pool_candidates])

        self.active_candidates, self.active_targets = pool_candidates, pool_targets
        self.active_seqs = pool_seqs

        pareto_candidates, pareto_targets = pareto_frontier(self.active_candidates, self.active_targets)
        pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pareto_candidates])
        pareto_cand_history = pareto_candidates.copy()
        pareto_seq_history = pareto_seqs.copy()
        pareto_target_history = pareto_targets.copy()
        norm_pareto_targets = hypercube_transform(pareto_targets)
        self._ref_point = -infer_reference_point(-torch.tensor(norm_pareto_targets)).numpy()
        print(self._ref_point)
        rescaled_ref_point = hypercube_transform.inv_transform(self._ref_point.copy())

        # logging setup
        total_bb_evals = 0
        start_time = time.time()
        round_idx = 0
        self._log_candidates(pareto_candidates, pareto_targets, round_idx, log_prefix)
        metrics = self._log_optimizer_metrics(norm_pareto_targets, round_idx, total_bb_evals, start_time, log_prefix)

        print('\n best candidates')
        obj_vals = {f'obj_val_{i}': pareto_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
        print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

        for round_idx in range(1, self.num_rounds + 1):
            metrics = {}

            # contract active pool to current Pareto frontier
            if (self.concentrate_pool > 0 and round_idx % self.concentrate_pool == 0) or self.latent_init == 'perturb_pareto':
                self.active_candidates, self.active_targets = pareto_frontier(
                    self.active_candidates, self.active_targets
                )
                self.active_seqs = np.array([a_cand.mutant_residue_seq for a_cand in self.active_candidates])
                print(f'\nactive set contracted to {self.active_candidates.shape[0]} pareto points')
            # augment active set with old pareto points
            if self.active_candidates.shape[0] < batch_size:
                num_samples = min(batch_size, pareto_cand_history.shape[0])
                num_backtrack = min(num_samples, batch_size - self.active_candidates.shape[0])
                _, weights, _ = weighted_resampling(pareto_target_history, k=self.resampling_weight)
                hist_idxs = np.random.choice(
                    np.arange(pareto_cand_history.shape[0]), num_samples, p=weights, replace=False
                )
                is_active = np.in1d(pareto_seq_history[hist_idxs], self.active_seqs)
                hist_idxs = hist_idxs[~is_active]
                if hist_idxs.size > 0:
                    hist_idxs = hist_idxs[:num_backtrack]
                    backtrack_candidates = pareto_cand_history[hist_idxs]
                    backtrack_targets = pareto_target_history[hist_idxs]
                    backtrack_seqs = pareto_seq_history[hist_idxs]
                    self.active_candidates = np.concatenate((self.active_candidates, backtrack_candidates))
                    self.active_targets = np.concatenate((self.active_targets, backtrack_targets))
                    self.active_seqs = np.concatenate((self.active_seqs, backtrack_seqs))
                    print(f'active set augmented with {backtrack_candidates.shape[0]} backtrack points')
            # augment active set with random points
            if self.active_candidates.shape[0] < batch_size:
                num_samples = min(batch_size, pool_candidates.shape[0])
                num_rand = min(num_samples, batch_size - self.active_candidates.shape[0])
                _, weights, _ = weighted_resampling(pool_targets, k=self.resampling_weight)
                rand_idxs = np.random.choice(
                    np.arange(pool_candidates.shape[0]), num_samples, p=weights, replace=False
                )
                is_active = np.in1d(pool_seqs[rand_idxs], self.active_seqs)
                rand_idxs = rand_idxs[~is_active][:num_rand]
                rand_candidates = pool_candidates[rand_idxs]
                rand_targets = pool_targets[rand_idxs]
                rand_seqs = pool_seqs[rand_idxs]
                self.active_candidates = np.concatenate((self.active_candidates, rand_candidates))
                self.active_targets = np.concatenate((self.active_targets, rand_targets))
                self.active_seqs = np.concatenate((self.active_seqs, rand_seqs))
                print(f'active set augmented with {rand_candidates.shape[0]} random points')

            print(rescaled_ref_point)
            print(self.active_targets)
            for seq in self.active_seqs:
                if hasattr(self.tokenizer, 'to_smiles'):
                    print(self.tokenizer.to_smiles(seq))
                else:
                    print(seq)

            print('\n---- fitting surrogate model ----')
            # acquisition fns assume maximization so we normalize and negate targets here
            z_score_transform = Normalizer(all_targets.mean(0), all_targets.std(0))

            tgt_transform = lambda x: -z_score_transform(x)
            transformed_ref_point = tgt_transform(rescaled_ref_point)

            new_split = DataSplit(new_seqs, new_targets)
            holdout_ratio = self.surrogate_model.holdout_ratio
            all_splits = update_splits(
                self.train_split, self.val_split, self.test_split, new_split, holdout_ratio,
            )
            self.train_split, self.val_split, self.test_split = all_splits

            X_train, Y_train = self.train_split.inputs, tgt_transform(self.train_split.targets)
            X_val, Y_val = self.val_split.inputs, tgt_transform(self.val_split.targets)
            X_test, Y_test = self.test_split.inputs, tgt_transform(self.test_split.targets)

            records = self.surrogate_model.fit(
                X_train, Y_train, X_val, Y_val, X_test, Y_test,
                encoder_obj=self.encoder_obj, resampling_temp=None
            )

            # log result
            last_entry = {key.split('/')[-1]: val for key, val in records[-1].items()}
            best_idx = last_entry['best_epoch']
            best_entry = {key.split('/')[-1]: val for key, val in records[best_idx].items()}
            print(pd.DataFrame([best_entry]).to_markdown(floatfmt='.4f'))
            metrics.update(dict(
                test_rmse=best_entry['test_rmse'],
                test_nll=best_entry['test_nll'],
                test_s_rho=best_entry['test_s_rho'],
                test_ece=best_entry['test_ece'],
                test_post_var=best_entry['test_post_var'],
                test_perplexity=best_entry['test_perplexity'],
                round_idx=round_idx,
                num_bb_evals=total_bb_evals,
                num_train=X_train.shape[0],
                time_elapsed=time.time() - start_time,
            ))
            metrics = {
                '/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()
            }
            wandb.log(metrics)

            baseline_seqs = np.array([cand.mutant_residue_seq for cand in self.active_candidates])
            baseline_targets = self.active_targets
            baseline_seqs, baseline_targets = pareto_frontier(baseline_seqs, baseline_targets)
            baseline_targets = tgt_transform(baseline_targets)

            acq_fn = hydra.utils.instantiate(
                self.acquisition,
                X_baseline=baseline_seqs,
                known_targets=torch.tensor(baseline_targets).to(self.surrogate_model.device),
                surrogate=self.surrogate_model,
                ref_point=torch.tensor(transformed_ref_point).to(self.surrogate_model.device),
                obj_dim=self.bb_task.obj_dim,
            )

            print('\n---- optimizing candidates ----')
            if self.resampling_weight is None:
                weights = np.ones(self.active_targets.shape[0]) / self.active_targets.shape[0]
            else:
                _, weights, _ = weighted_resampling(self.active_targets, k=self.resampling_weight)

            base_cand_batches = []
            new_seq_batches = []
            new_seq_scores = []
            batch_entropy = []
            for gen_idx in range(self.num_gens):
                # select candidate sequences to mutate
                base_idxs = np.random.choice(np.arange(weights.shape[0]), batch_size, p=weights, replace=True)
                base_candidates = self.active_candidates[base_idxs]
                base_seqs = np.array([cand.mutant_residue_seq for cand in base_candidates])
                base_tok_idxs = str_to_tokens(base_seqs, self.encoder.tokenizer)
                base_mask = (base_tok_idxs != self.encoder.tokenizer.padding_idx)
                base_lens = base_mask.float().sum(-1).long()
                tgt_lens = None if self.bb_task.allow_len_change else base_lens

                with torch.no_grad():
                    window_mask_idxs, window_entropy = evaluate_windows(
                        base_seqs, self.encoder, self.window_size, replacement=True, encoder_obj=self.encoder_obj
                    )

                # select token positions to mutate
                if self.position_sampler == 'entropy_method':
                    mask_idxs = self.sample_mutation_window(window_mask_idxs, window_entropy)
                elif self.position_sampler == 'uniform':
                    mask_idxs = np.concatenate([
                        random.choice(w_idxs) for w_idxs in window_mask_idxs.values()
                    ])
                else:
                    raise ValueError

                with torch.no_grad():
                    src_tok_idxs = base_tok_idxs.clone().to(self.surrogate_model.device)
                    if self.latent_init == 'perturb_pareto':
                        opt_features, src_mask = self.encoder.get_token_features(src_tok_idxs)
                        opt_features += 1e-3 * torch.randn_like(opt_features)
                    elif self.encoder_obj == 'lanmt':
                        src_tok_idxs = corrupt_tok_idxs(
                            src_tok_idxs, self.encoder.tokenizer, max_len_delta=None, select_idxs=mask_idxs
                        )
                        opt_features, src_mask = self.encoder.get_token_features(src_tok_idxs)
                    elif self.encoder_obj == 'mlm':
                        # this line assumes padding tokens are always added at the end
                        np.put_along_axis(src_tok_idxs, mask_idxs, self.encoder.tokenizer.masking_idx, axis=1)
                        src_tok_features, src_mask = self.encoder.get_token_features(src_tok_idxs)
                        opt_features = np.take_along_axis(src_tok_features, mask_idxs[..., None], axis=1)
                    else:
                        raise ValueError

                    # initialize latent token-choice decision variables
                    opt_params = torch.empty(
                        *opt_features.shape, requires_grad=self.optimize_latent, device=self.surrogate_model.device,
                        dtype=self.surrogate_model.dtype
                    )
                    opt_params.copy_(opt_features)

                # optimize decision variables
                optimizer = torch.optim.Adam(params=[opt_params], lr=self.lr, betas=(0., 1e-2))
                lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience)
                best_score, best_step = None, 0
                for step_idx in range(self.num_opt_steps):
                    if self.encoder_obj == 'lanmt':
                        lat_tok_features, pooled_features = self.encoder.pool_features(opt_params, src_mask)
                        tgt_tok_logits, tgt_mask = self.encoder.logits_from_features(
                            opt_params, src_mask, lat_tok_features, tgt_lens
                        )
                        tgt_tok_idxs, logit_entropy = self.encoder.sample_tgt_tok_idxs(
                            tgt_tok_logits, tgt_mask, temp=1.
                        )
                    elif self.encoder_obj == 'mlm':
                        current_features = src_tok_features.clone()
                        np.put_along_axis(current_features, mask_idxs[..., None], opt_params, axis=1)
                        lat_tok_features, pooled_features = self.encoder.pool_features(current_features, src_mask)
                        tgt_tok_logits, tgt_mask = self.encoder.logits_from_features(
                            current_features, src_mask, lat_tok_features, tgt_lens
                        )
                        new_tok_idxs, logit_entropy = sample_tokens(
                            base_tok_idxs, tgt_tok_logits, self.encoder.tokenizer, replacement=False
                        )
                        new_tok_idxs = np.take_along_axis(new_tok_idxs, mask_idxs, axis=1)
                        tgt_tok_idxs = src_tok_idxs.clone()
                        np.put_along_axis(tgt_tok_idxs, mask_idxs, new_tok_idxs, axis=1)
                        logit_entropy = np.take_along_axis(logit_entropy, mask_idxs, axis=1)
                    else:
                        raise ValueError

                    lat_acq_vals = acq_fn(pooled_features.unsqueeze(0))
                    loss = -lat_acq_vals.mean() + self.entropy_penalty * logit_entropy.mean()

                    if self.optimize_latent:
                        loss.backward()
                        optimizer.step()
                        lr_sched.step(loss)

                    tgt_seqs = tokens_to_str(tgt_tok_idxs, self.encoder.tokenizer)
                    act_acq_vals = acq_fn(tgt_seqs[None, :]).mean().item()

                    best_score, best_step, _, stop = check_early_stopping(
                        model=None,
                        best_score=best_score,
                        best_epoch=best_step,
                        best_weights=None,
                        curr_score=-act_acq_vals,
                        curr_epoch=step_idx + 1,
                        patience=self.patience,
                        save_weights=False,
                    )
                    if (step_idx + 1) == best_step:
                        best_seqs = tgt_seqs.copy()
                        best_entropy = logit_entropy.mean().item()
                    if stop:
                        break

                base_cand_batches.append(base_candidates.copy())
                new_seq_batches.append(best_seqs.copy())
                new_seq_scores.append(best_score)
                batch_entropy.append(best_entropy)

                # print(f'batch {gen_idx + 1}: score - {best_score:0.4f}, entropy - {logit_entropy.mean().item():0.4f}')

            # score all decoded batches, observe the highest value batch
            new_seq_batches = np.stack(new_seq_batches)
            new_seq_scores = np.stack(new_seq_scores)
            best_batch_idx = new_seq_scores.argmin()

            base_candidates = base_cand_batches[best_batch_idx]
            base_seqs = np.array([b_cand.mutant_residue_seq for b_cand in base_candidates])
            new_seqs = new_seq_batches[best_batch_idx]
            # new_tokens = new_tok_batches[best_batch_idx]

            # logging
            metrics = dict(
                acq_val=new_seq_scores[best_batch_idx].mean().item(),
                entropy=batch_entropy[best_batch_idx],
                round_idx=round_idx,
                num_bb_evals=total_bb_evals,
                time_elapsed=time.time() - start_time,
            )
            print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
            metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
            wandb.log(metrics)

            print('\n---- querying objective function ----')
            new_candidates = self.bb_task.make_new_candidates(base_candidates, new_seqs)

            # filter infeasible candidates
            is_feasible = self.bb_task.is_feasible(new_candidates)
            base_candidates = base_candidates[is_feasible]
            base_seqs = base_seqs[is_feasible]
            new_seqs = new_seqs[is_feasible]
            new_candidates = new_candidates[is_feasible]
            # new_tokens = new_tokens[is_feasible]
            if new_candidates.size == 0:
                print('no new candidates')
                continue

            # filter duplicate candidates
            new_seqs, unique_idxs = np.unique(new_seqs, return_index=True)
            base_candidates = base_candidates[unique_idxs]
            base_seqs = base_seqs[unique_idxs]
            new_candidates = new_candidates[unique_idxs]

            # filter redundant candidates
            is_new = np.in1d(new_seqs, all_seqs, invert=True)
            base_candidates = base_candidates[is_new]
            base_seqs = base_seqs[is_new]
            new_seqs = new_seqs[is_new]
            new_candidates = new_candidates[is_new]
            if new_candidates.size == 0:
                print('no new candidates')
                continue

            new_targets = self.bb_task.score(new_candidates)
            all_targets = np.concatenate((all_targets, new_targets))
            all_seqs = np.concatenate((all_seqs, new_seqs))

            for seq in new_seqs:
                if hasattr(self.tokenizer, 'to_smiles'):
                    print(self.tokenizer.to_smiles(seq))
                else:
                    print(seq)

            assert base_seqs.shape[0] == new_seqs.shape[0] and new_seqs.shape[0] == new_targets.shape[0]
            for b_cand, n_cand, f_val in zip(base_candidates, new_candidates, new_targets):
                print(f'{len(b_cand)} --> {len(n_cand)}: {f_val}')

            pool_candidates = np.concatenate((pool_candidates, new_candidates))
            pool_targets = np.concatenate((pool_targets, new_targets))
            pool_seqs = np.concatenate((pool_seqs, new_seqs))

            # augment active pool with candidates that can be mutated again
            self.active_candidates = np.concatenate((self.active_candidates, new_candidates))
            self.active_targets = np.concatenate((self.active_targets, new_targets))
            self.active_seqs = np.concatenate((self.active_seqs, new_seqs))

            # overall Pareto frontier including terminal candidates
            pareto_candidates, pareto_targets = pareto_frontier(
                np.concatenate((pareto_candidates, new_candidates)),
                np.concatenate((pareto_targets, new_targets)),
            )
            pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pareto_candidates])

            print('\n new candidates')
            obj_vals = {f'obj_val_{i}': new_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
            print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

            print('\n best candidates')
            obj_vals = {f'obj_val_{i}': pareto_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
            print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

            # store good candidates for backtracking
            par_is_new = np.in1d(pareto_seqs, pareto_seq_history, invert=True)
            pareto_cand_history = safe_np_cat([pareto_cand_history, pareto_candidates[par_is_new]])
            pareto_seq_history = safe_np_cat([pareto_seq_history, pareto_seqs[par_is_new]])
            pareto_target_history = safe_np_cat([pareto_target_history, pareto_targets[par_is_new]])

            # logging
            norm_pareto_targets = hypercube_transform(pareto_targets)
            total_bb_evals += batch_size
            self._log_candidates(new_candidates, new_targets, round_idx, log_prefix)
            metrics = self._log_optimizer_metrics(
                norm_pareto_targets, round_idx, total_bb_evals, start_time, log_prefix
            )

        return metrics

    def sample_mutation_window(self, window_mask_idxs, window_entropy, temp=1.):
        # selected_features = []
        selected_mask_idxs = []
        for seq_idx, entropies in window_entropy.items():
            mask_idxs = window_mask_idxs[seq_idx]
            assert len(mask_idxs) == len(entropies)
            window_idxs = np.arange(len(mask_idxs)).astype(int)
            entropies = torch.tensor(entropies)
            weights = F.softmax(entropies / temp).cpu().numpy()
            selected_window = np.random.choice(window_idxs, 1, p=weights).item()
            selected_mask_idxs.append(mask_idxs[selected_window])
        return np.concatenate(selected_mask_idxs)

    def _log_candidates(self, candidates, targets, round_idx, log_prefix):
        table_cols = ['round_idx', 'cand_uuid', 'cand_ancestor', 'cand_seq']
        table_cols.extend([f'obj_val_{idx}' for idx in range(self.bb_task.obj_dim)])
        for cand, obj in zip(candidates, targets):
            new_row = [round_idx, cand.uuid, cand.wild_name, cand.mutant_residue_seq]
            new_row.extend([elem for elem in obj])
            record = {'/'.join((log_prefix, 'candidates', key)): val for key, val in zip(table_cols, new_row)}
            wandb.log(record)

    def _log_optimizer_metrics(self, normed_targets, round_idx, num_bb_evals, start_time, log_prefix):
        hv_indicator = get_performance_indicator('hv', ref_point=self._ref_point)
        new_hypervol = hv_indicator.do(normed_targets)
        self._hv_ref = new_hypervol if self._hv_ref is None else self._hv_ref
        metrics = dict(
            round_idx=round_idx,
            hypervol_abs=new_hypervol,
            hypervol_rel=new_hypervol / max(1e-6, self._hv_ref),
            num_bb_evals=num_bb_evals,
            time_elapsed=time.time() - start_time,
        )
        print(pd.DataFrame([metrics]).to_markdown(floatfmt='.4f'))
        metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
        wandb.log(metrics)
        return metrics
