import numpy as np
import torch
import hydra
import wandb
import time
import pandas as pd

from botorch.utils.multi_objective import pareto, infer_reference_point

from pymoo.factory import get_termination, get_performance_indicator
from pymoo.optimize import minimize

from lambo.tasks.surrogate_task import SurrogateTask
from lambo.models.lm_elements import LanguageModel
from lambo.utils import weighted_resampling, DataSplit, update_splits, safe_np_cat


def pareto_frontier(candidate_pool, obj_vals):
    """
    args:
        candidate_pool: NumPy array of candidate objects
        obj_vals: NumPy array of objective values (assumes minimization)
    """
    assert len(candidate_pool) == obj_vals.shape[0]
    if len(candidate_pool) == 1:
        return candidate_pool, obj_vals
    # pareto utility assumes maximization
    pareto_mask = pareto.is_non_dominated(-torch.tensor(obj_vals))
    return candidate_pool[pareto_mask], obj_vals[pareto_mask]


class Normalizer(object):
    def __init__(self, loc=0., scale=1.):
        self.loc = loc
        self.scale = np.where(scale != 0, scale, 1.)

    def __call__(self, arr):
        return (arr - self.loc) / self.scale

    def inv_transform(self, arr):
        return self.scale * arr + self.loc


class SequentialGeneticOptimizer(object):
    def __init__(self, bb_task, algorithm, tokenizer, num_rounds, num_gens, seed, concentrate_pool=1,
                 residue_sampler='uniform', resampling_weight=1., **kwargs):
        self.bb_task = bb_task
        self.algorithm = algorithm
        self.num_rounds = num_rounds
        self.num_gens = num_gens
        self.term_fn = get_termination("n_gen", num_gens)
        self.seed = seed
        self.concentrate_pool = concentrate_pool
        self.residue_sampler = residue_sampler

        tokenizer.set_sampling_vocab(None, bb_task.max_ngram_size)
        self.tokenizer = tokenizer

        self.encoder = None

        self._hv_ref = None
        self._ref_point = np.array([1] * self.bb_task.obj_dim)

        self.active_candidates = None
        self.active_targets = None
        self.resampling_weight = resampling_weight

    def optimize(self, candidate_pool, pool_targets, all_seqs, all_targets, log_prefix=''):
        batch_size = self.bb_task.batch_size
        target_min = all_targets.min(axis=0)
        target_range = all_targets.max(axis=0) - target_min
        hypercube_transform = Normalizer(
            loc=target_min + 0.5 * target_range,
            scale=target_range / 2.,
        )

        bb_task = hydra.utils.instantiate(self.bb_task, tokenizer=self.tokenizer, candidate_pool=candidate_pool,
                                          batch_size=1)
        is_feasible = bb_task.is_feasible(candidate_pool)
        pool_candidates = candidate_pool[is_feasible]
        pool_targets = pool_targets[is_feasible]
        pool_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pool_candidates])

        self.all_seqs = all_seqs
        self.all_targets = all_targets
        new_seqs = all_seqs.copy()
        new_targets = all_targets.copy()
        self.active_candidates, self.active_targets = pool_candidates, pool_targets
        self.active_seqs = pool_seqs

        pareto_candidates, pareto_targets = pareto_frontier(self.active_candidates, self.active_targets)
        self.pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pareto_candidates])
        pareto_cand_history = pareto_candidates.copy()
        pareto_seq_history = self.pareto_seqs.copy()
        pareto_target_history = pareto_targets.copy()
        norm_pareto_targets = hypercube_transform(pareto_targets)
        self._ref_point = -infer_reference_point(-torch.tensor(norm_pareto_targets)).numpy()
        rescaled_ref_point = hypercube_transform.inv_transform(self._ref_point)

        # logging setup
        total_bb_evals = 0
        start_time = time.time()
        round_idx = 0
        self._log_candidates(pareto_candidates, pareto_targets, round_idx, log_prefix)
        metrics = self._log_optimizer_metrics(norm_pareto_targets, round_idx, total_bb_evals, start_time, log_prefix)

        print('\n best candidates')
        obj_vals = {f'obj_val_{i}': pareto_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
        print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

        # set up encoder which may also be a masked language model (MLM)
        encoder = None if self.encoder is None else hydra.utils.instantiate(
            self.encoder, tokenizer=self.tokenizer
        )

        if self.residue_sampler == 'uniform':
            mlm_obj = None
        elif self.residue_sampler == 'mlm':
            assert isinstance(encoder, LanguageModel)
            mlm_obj = encoder
        else:
            raise ValueError

        for round_idx in range(1, self.num_rounds + 1):
            # contract active pool to current Pareto frontier
            if self.concentrate_pool > 0 and round_idx % self.concentrate_pool == 0:
                self.active_candidates, self.active_targets = pareto_frontier(self.active_candidates, self.active_targets)
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

            if self.resampling_weight is None:
                active_weights = np.ones(self.active_targets.shape[0]) / self.active_targets.shape[0]
            else:
                _, active_weights, _ = weighted_resampling(self.active_targets, k=self.resampling_weight)

            # prepare the inner task
            z_score_transform = Normalizer(self.all_targets.mean(0), self.all_targets.std(0))

            # algorithm setup
            algorithm = hydra.utils.instantiate(self.algorithm)
            algorithm.initialization.sampling.tokenizer = self.tokenizer
            algorithm.mating.mutation.tokenizer = self.tokenizer

            if not self.residue_sampler == 'uniform':
                algorithm.initialization.sampling.mlm_obj = mlm_obj
                algorithm.mating.mutation.mlm_obj = mlm_obj

            problem = self._create_inner_task(
                candidate_pool=self.active_candidates,
                candidate_weights=active_weights,
                input_data=new_seqs,
                target_data=new_targets,
                transform=z_score_transform,
                ref_point=rescaled_ref_point,
                encoder=encoder,
                round_idx=round_idx,
                num_bb_evals=total_bb_evals,
                start_time=start_time,
                log_prefix=log_prefix,
            )

            print('---- optimizing candidates ----')
            res = minimize(
                problem,
                algorithm,
                self.term_fn,
                save_history=False,
                verbose=True
            )

            # query outer task, append data
            new_candidates, new_targets, new_seqs, bb_evals = self._evaluate_result(
                res, self.active_candidates, z_score_transform, round_idx, total_bb_evals, start_time, log_prefix
            )
            total_bb_evals += bb_evals

            # filter infeasible candidates
            is_feasible = bb_task.is_feasible(new_candidates)
            new_seqs = new_seqs[is_feasible]
            new_candidates = new_candidates[is_feasible]
            if new_candidates.size == 0:
                print('no new candidates')
                continue

            # filter duplicate candidates
            new_seqs, unique_idxs = np.unique(new_seqs, return_index=True)
            new_candidates = new_candidates[unique_idxs]
            new_targets = new_targets[unique_idxs]

            # filter redundant candidates
            is_new = np.in1d(new_seqs, self.all_seqs, invert=True)
            new_seqs = new_seqs[is_new]
            new_candidates = new_candidates[is_new]
            new_targets = new_targets[is_new]
            if new_candidates.size == 0:
                print('no new candidates')
                self._log_optimizer_metrics(
                    norm_pareto_targets, round_idx, total_bb_evals, start_time, log_prefix
                )
                continue

            pool_candidates = np.concatenate((pool_candidates, new_candidates))
            pool_targets = np.concatenate((pool_targets, new_targets))
            pool_seqs = np.concatenate((pool_seqs, new_seqs))

            self.all_seqs = np.concatenate((self.all_seqs, new_seqs))
            self.all_targets = np.concatenate((self.all_targets, new_targets))

            for seq in new_seqs:
                if hasattr(self.tokenizer, 'to_smiles'):
                    print(self.tokenizer.to_smiles(seq))
                else:
                    print(seq)

            # augment active pool with candidates that can be mutated again
            self.active_candidates = np.concatenate((self.active_candidates, new_candidates))
            self.active_targets = np.concatenate((self.active_targets, new_targets))
            self.active_seqs = np.concatenate((self.active_seqs, new_seqs))

            # overall Pareto frontier including terminal candidates
            pareto_candidates, pareto_targets = pareto_frontier(
                np.concatenate((pareto_candidates, new_candidates)),
                np.concatenate((pareto_targets, new_targets)),
            )
            self.pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pareto_candidates])

            print(new_targets)
            print('\n new candidates')
            obj_vals = {f'obj_val_{i}': new_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
            print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

            print('\n best candidates')
            obj_vals = {f'obj_val_{i}': pareto_targets[:, i].min() for i in range(self.bb_task.obj_dim)}
            print(pd.DataFrame([obj_vals]).to_markdown(floatfmt='.4f'))

            par_is_new = np.in1d(self.pareto_seqs, pareto_seq_history, invert=True)
            pareto_cand_history = safe_np_cat([pareto_cand_history, pareto_candidates[par_is_new]])
            pareto_seq_history = safe_np_cat([pareto_seq_history, self.pareto_seqs[par_is_new]])
            pareto_target_history = safe_np_cat([pareto_target_history, pareto_targets[par_is_new]])

            # logging
            norm_pareto_targets = hypercube_transform(pareto_targets)
            self._log_candidates(new_candidates, new_targets, round_idx, log_prefix)
            metrics = self._log_optimizer_metrics(norm_pareto_targets, round_idx, total_bb_evals, start_time, log_prefix)

        return metrics

    def _evaluate_result(self, *args, **kwargs):
        raise NotImplementedError

    def _create_inner_task(self, *args, **kwargs):
        raise NotImplementedError

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
        print(pd.DataFrame([metrics]).to_markdown())
        metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
        wandb.log(metrics)
        return metrics


class ModelFreeGeneticOptimizer(SequentialGeneticOptimizer):
    def _create_inner_task(
            self, candidate_pool, input_data, target_data, transform, candidate_weights, *args, **kwargs):
        inner_task = hydra.utils.instantiate(
            self.bb_task,
            candidate_pool=candidate_pool,
            transform=transform,
            tokenizer=self.tokenizer,
            batch_size=1,
            candidate_weights=candidate_weights,
        )
        return inner_task

    def _evaluate_result(self, result, candidate_pool, transform, *args, **kwargs):
        new_candidates = result.pop.get('X_cand').reshape(-1)
        new_seqs = result.pop.get('X_seq').reshape(-1)
        new_targets = transform.inv_transform(result.pop.get('F'))
        bb_evals = self.num_gens * self.algorithm.pop_size
        return new_candidates, new_targets, new_seqs, bb_evals


class ModelBasedGeneticOptimizer(SequentialGeneticOptimizer):
    def __init__(
            self, bb_task, surrogate, algorithm, acquisition, encoder, tokenizer, num_rounds, num_gens, seed,
            encoder_obj, **kwargs
    ):
        super().__init__(
            bb_task=bb_task,
            algorithm=algorithm,
            tokenizer=tokenizer,
            num_rounds=num_rounds,
            num_gens=num_gens,
            seed=seed,
            **kwargs
        )
        self.encoder = encoder
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.surrogate_model = None
        self.train_split = DataSplit()
        self.val_split = DataSplit()
        self.test_split = DataSplit()
        self.encoder_obj = encoder_obj

    def _create_inner_task(self, candidate_pool, candidate_weights, input_data, target_data, transform, ref_point,
                           encoder, round_idx, num_bb_evals, start_time, log_prefix):

        if self.surrogate_model is None:
            self.surrogate_model = hydra.utils.instantiate(self.surrogate, encoder=encoder, tokenizer=encoder.tokenizer,
                                                           alphabet=self.tokenizer.non_special_vocab)

        # prepare surrogate dataset
        tgt_transform = lambda x: -transform(x)
        transformed_ref_point = tgt_transform(ref_point)

        new_split = DataSplit(input_data, target_data)
        holdout_ratio = self.surrogate.holdout_ratio
        all_splits = update_splits(
            self.train_split, self.val_split, self.test_split, new_split, holdout_ratio,
        )
        self.train_split, self.val_split, self.test_split = all_splits

        X_train, Y_train = self.train_split.inputs, tgt_transform(self.train_split.targets)
        X_val, Y_val = self.val_split.inputs, tgt_transform(self.val_split.targets)
        X_test, Y_test = self.test_split.inputs, tgt_transform(self.test_split.targets)

        # train surrogate
        records = self.surrogate_model.fit(
            X_train, Y_train, X_val, Y_val, X_test, Y_test, resampling_temp=None,
            encoder_obj=self.encoder_obj
        )
        # log result
        last_entry = {key.split('/')[-1]: val for key, val in records[-1].items()}
        best_idx = last_entry['best_epoch']
        best_entry = {key.split('/')[-1]: val for key, val in records[best_idx].items()}
        print(pd.DataFrame([best_entry]).to_markdown())
        metrics = dict(
            test_rmse=best_entry['test_rmse'],
            test_nll=best_entry['test_nll'],
            test_s_rho=best_entry['test_s_rho'],
            test_ece=best_entry['test_ece'],
            test_post_var=best_entry['test_post_var'],
            round_idx=round_idx,
            num_bb_evals=num_bb_evals,
            num_train=self.train_split.inputs.shape[0],
            time_elapsed=time.time() - start_time,
        )
        metrics = {
            '/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()
        }
        wandb.log(metrics)

        # complete task setup
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
        inner_task = SurrogateTask(self.tokenizer, candidate_pool, acq_fn, batch_size=acq_fn.batch_size)

        return inner_task

    def _evaluate_result(self, result, candidate_pool, transform, round_idx, num_bb_evals, start_time, log_prefix,
                         *args, **kwargs):
        all_x = result.pop.get('X')
        all_acq_vals = result.pop.get('F')

        cand_batches = result.problem.x_to_query_batches(all_x)
        query_points = cand_batches[0]
        query_acq_vals = all_acq_vals[0]

        batch_idx = 1
        while query_points.shape[0] < self.bb_task.batch_size:
            query_points = np.concatenate((query_points, cand_batches[batch_idx]))
            query_acq_vals = np.concatenate((query_acq_vals, all_acq_vals[batch_idx]))
            batch_idx += 1

        bb_task = hydra.utils.instantiate(
            self.bb_task, tokenizer=self.tokenizer, candidate_pool=candidate_pool, batch_size=1
        )
        bb_out = bb_task.evaluate(query_points, return_as_dictionary=True)
        new_candidates = bb_out['X_cand'].reshape(-1)
        new_seqs = bb_out['X_seq'].reshape(-1)
        new_targets = bb_out["F"]
        bb_evals = query_points.shape[0]

        metrics = dict(
            acq_val=query_acq_vals.mean().item(),
            round_idx=round_idx,
            num_bb_evals=num_bb_evals,
            time_elapsed=time.time() - start_time,
        )
        metrics = {'/'.join((log_prefix, 'opt_metrics', key)): val for key, val in metrics.items()}
        wandb.log(metrics)

        return new_candidates, new_targets, new_seqs, bb_evals
