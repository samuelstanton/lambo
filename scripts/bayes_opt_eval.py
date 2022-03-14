import os
import h5py
import tqdm
import torch
import wandb
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import bo_protein.utils
from bo_protein.models import surrogates
from bo_protein.gfp_data import utils
from bo_protein import rewards, acquisition, optimizers
from regression import ind_splits

import warnings
warnings.filterwarnings("ignore")


def _normalize(Y):
    Y_mean, Y_std = np.mean(Y), np.std(Y)
    Y = (Y - Y_mean) / np.where(Y_std != 0, Y_std, 1)
    return Y


class BayesOptRanking:

    def __init__(self, acq_func, X, Y):
        self.acq_func = acq_func
        self.X, self.Y = X, Y
        self.num_examples = len(X)

    def pick_start_examples(self, seed, num_start=500, save_perc=0.0):
        np.random.seed(seed)
        start_idxs = np.random.choice(self.num_examples, size=num_start)
        search_idxs = np.setdiff1d(np.arange(self.num_examples), start_idxs)
        return start_idxs, search_idxs

    def benchmark_random(self, search_idxs, num_iter=20, top_k=100):
        rewards = []
        for i in range(num_iter):
            idx = np.random.choice(len(search_idxs), size=top_k)
            random_idxs = search_idxs[idx]
            Y_best = self.Y[random_idxs]
            
            # print(Y_best)
            # print(np.max(Y_best))

            mask = np.ones(len(search_idxs), dtype=bool)
            mask[idx] = False
            search_idxs = search_idxs[mask]

            rewards.append(np.max(Y_best))
        rewards = np.array(rewards)
        return rewards

    def run(self, seed, num_iter=20, top_k=100, random_only=False, tag=""):
        start_idxs, search_idxs = self.pick_start_examples(seed)

        if random_only:
            return self.benchmark_random(search_idxs, num_iter=num_iter, top_k=top_k)

        X_start, Y_start = self.X[start_idxs], self.Y[start_idxs]
        X_search, Y_search = self.X[search_idxs], self.Y[search_idxs]

        scores = []
        proposals = []
        for it in tqdm.tqdm(range(num_iter)):
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_start, Y_start, test_size=0.2
            )

            self.acq_func.surrogate.fit(
                X_train,
                Y_train,
                X_test,
                Y_test,
                log_prefix=f"{tag}_rank_acq/iter_{it}",
            )

            with torch.no_grad():
                labels = self.acq_func.score(X_search)
        
            max_idx = np.argsort(-labels)[:top_k]

            X_best, Y_best = X_search[max_idx], Y_search[max_idx]

            X_start = np.concatenate([X_start, X_best], axis=0)
            Y_start = np.concatenate([Y_start, Y_best], axis=0)

            mask = np.ones(len(X_search), dtype=bool)
            mask[max_idx] = False
            X_search = X_search[mask]
            Y_search = Y_search[mask]
            
            try:
                wandb.log({'iter_max_obj': np.max(Y_best), 'cum_max_obj': np.max(Y_start)})
            except Exception as e:
                print(e)

            scores.append(Y_best)
            proposals.append(X_best)

        return scores, proposals


class BayesOptTask:
    def __init__(self, reward_func, acq_func, optimizer, config):
        noise_variance = config.get('reward_noise_var', 0.)
        if noise_variance > 0:
            noise_model = acquisition.HomoskedasticNoise(noise_variance)
            self.reward_func = rewards.NoisyReward(reward_func, noise_model)
        else:
            self.reward_func = reward_func
        self.acq_func = acq_func
        self.opt = optimizer
        self.config = config

    def benchmark_genetic(self, X, num_iter=10, top_k=100, tag=""):
        pop_size = self.opt.population_size
        num_evolutions = self.opt.num_evolutions

        self.opt.population_size = top_k
        self.opt.num_evolutions = num_iter

        *results, logs = self.opt.optimize(
            X,
            self.reward_func,
            top_k=top_k,
            per_iter_log=True,
            log_prefix=f"{tag}_genetic_opt",
        )

        results = []
        for it, (scores, seqs) in enumerate(zip(logs[0], logs[1])):
            results += [[it * top_k, idx, x, y] for idx, (x, y) in enumerate(zip(seqs, scores))]

        self.opt.population_size = pop_size
        self.opt.num_evolutions = num_evolutions

        return results

    def run(self, X, num_iter=2, top_k=100, genetic_only=False, tag=""):
        if genetic_only:
            return self.benchmark_genetic(X, num_iter=num_iter, top_k=top_k, tag=tag)

        Y = self.reward_func.score(X)
        if hasattr(self.acq_func, 'observe'):
            self.acq_func.observe(X)

        results = [[0, idx, x, y] for idx, (x, y) in enumerate(zip(X, Y))]
        eval_ratio = 0.15
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=eval_ratio)
        num_obj_queries = 0

        # records = [dict(seq=x, label=y, split='train', batch=0) for x, y in zip(X_train, Y_train)]
        # records.extend([dict(seq=x, label=y, split='test', batch=0) for x, y in zip(X_test, Y_test)])

        # with profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], profile_memory=True) as prof:
        for it in tqdm.tqdm(range(num_iter)):
            # print(f"***** iteration {it} *****")
            
            # renormalize data
            target_mean, target_std = Y_train.mean(0), Y_train.std(0)
            train_targets = (Y_train - target_mean) / np.where(target_std != 0., target_std, 1.)
            test_targets = (Y_test - target_mean) / np.where(target_std != 0., target_std, 1.) 

            # if self.config.get("log_shit", None) is not None:
            # np.save(f'notebooks/gp_shift_analysis/target_mean_{it}.npy', target_mean)
            # np.save(f'notebooks/gp_shift_analysis/target_std_{it}.npy', target_std)

            # with profiler.record_function('fit_surrogate'):
            # print('[BayesOpt] fitting surrogate')
            self.acq_func.surrogate.fit(
                X_train, train_targets, X_test, test_targets, reset=(it == 0)
            )

            if isinstance(self.opt, optimizers.DiscreteMaxPosteriorSampling):
                opt_args = [X_train, train_targets, self.acq_func.surrogate]
                opt_kwargs = dict(num_samples=top_k)
            else:
                opt_args = [X_train, self.acq_func]
                opt_kwargs = dict(top_k=top_k, pool_scores=Y_train, it=it)
            # with profiler.record_function('optimize_seqs'):
            # print('[BayesOpt] optimizing sequences')
            max_seq, max_targets, *other = self.opt.optimize(*opt_args, **opt_kwargs)
            del opt_args  # important to limit memory use

            print(max_seq)

            # post-process queries
            _, idx = np.unique(max_seq, return_index=True)
            max_seq, max_targets = max_seq[idx], max_targets[idx]
            max_f = max_targets * target_std + target_mean

            # label query points
            labels = self.reward_func.score(max_seq)
            labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels

            # set aside some for evaluation
            test_mask = np.random.rand(max_seq.shape[0]) < eval_ratio  # hold some data out for evaluation
            X_test = np.concatenate([X_test, max_seq[test_mask]])
            Y_test = np.concatenate([Y_test, labels[test_mask]])

            # records.extend([dict(
            #     seq=x, label=y, split='test', batch=it + 1
            # ) for x, y in zip(max_seq[test_mask], labels[test_mask])])

            # append to trainset and de-duplicate
            max_seq, labels = max_seq[~test_mask], labels[~test_mask]
            X_train = np.concatenate([X_train, max_seq])
            Y_train = np.concatenate([Y_train, labels])
            
            _, idx = np.unique(X_train, return_index=True)
            X_train = X_train[idx]
            Y_train = Y_train[idx]

            # print(f"{len(labels)}/{len(max_seq)}")
            print(labels)

            # if hasattr(self.acq_func, 'observe'):
            #     self.acq_func.observe(max_seq)  # updates obs counts for multi-fidelity

            # logging
            num_obj_queries += max_seq.shape[0]
            metrics = {
                "f_sample_mean": max_f.mean(),
                "f_sample_max": max_f.max(),
                "f_sample_min": max_f.min(),
                'f_true_mean': labels.mean(),
                'f_true_max': labels.max(),
                'f_true_min': labels.min(),
                'f_best': np.max(Y_train),
                'num_obj_queries': num_obj_queries,
                "num_seen": X_train.shape[0],
                'bo_iter': it + 1,
            }
            try:
                wandb.log({'iter_max_obj': np.max(labels), 'cum_max_obj': np.max(Y_train)})
                wandb.log(metrics)
            except Exception as e:
                print(e)

            results += [[num_obj_queries, idx, x, y] for idx, (x, y) in enumerate(zip(max_seq, labels))]

            # records.extend([dict(seq=x, label=y, split='train', batch=it + 1) for x, y in zip(max_seq, labels)])
            # print(prof.key_averages().table(sort_by="self_gpu_memory_usage", row_limit=10))

        # history = pd.DataFrame(records)
        # print(history[['label', 'batch']].to_markdown())
        # history.to_csv('notebooks/bayes_opt_history.csv', index=False)

        return results

def prep_data(source, task, surrogate, opt, seed):
    data, max_len, tag = ind_splits(source, task, seed=seed)
    X = np.concatenate([data[0], data[2]])
    Y = np.concatenate([data[1], data[3]])

    if opt:
        opt.max_len = max_len
    if surrogate and hasattr(surrogate, 'max_len'):
        surrogate.max_len = max_len 

    return data, X, Y, tag


def get_reward_func(config, reward, data, source, task, start_pool, seed):
    if reward == "rank":
        X = np.concatenate([data[0], data[2]])
        reward_func = rewards.MostCommonBigramRewardFunc(X)
    elif reward == "regex":
        X = np.concatenate([data[0], data[2]])
        reward_func = rewards.MostCommonBigramRewardFunc(X)
    elif reward == "random_nn":
        reward_config = {"ensemble_size": 3, "num_epochs": 0}
        save_fn = f"ckpts/reward_random_{source}_{task}.pt"
        reward_func = rewards.NNRewardFunc(data, reward_config, save_fn=save_fn)
    elif reward == "trained_nn":
        reward_config = {"ensemble_size": 3, "num_epochs": 10}
        save_fn = f"ckpts/reward_trained_{source}_{task}.pt"
        reward_func = rewards.NNRewardFunc(data, reward_config, save_fn=save_fn)
    elif reward == "hmm":
        reward_config = {"n_hidden": 10}
        save_fn = f"ckpts/reward_hmm_{source}_{task}.pkl"
        reward_func = rewards.HMMRewardFunc(data, reward_config, save_fn=save_fn)
    elif reward == "embed_dist":
        reward_func = rewards.EmbedDistRewardFunc({})
    elif reward == "rbf_kernel":
        X = np.concatenate([data[0], data[2]])
        non_start = np.setdiff1d(X, start_pool)
        np.random.seed(seed)
        prototype = np.random.choice(non_start, size=1)
        print(f"************ prototype: {prototype} *************")
        reward_func = rewards.RBFKernelRewardFunc(config, prototype_seq=prototype)
    elif reward == "rnafold":
        reward_func = rewards.RNAfoldRewardFunc({})

    return reward_func


def get_start_pool(config, X, seed):
    np.random.seed(seed)
    start_pool_size = min(len(X), config.get("start_pool_size", 500))
    start_pool = np.random.choice(X, size=start_pool_size, replace=False)
    return start_pool


def run_trial(config, reward, source, task, acq_func, opt, genetic_only, seed=0):
    data, X, Y, tag = prep_data(source, task, acq_func.surrogate, opt, seed)
    
    if reward == "rank":
        task = BayesOptRanking(acq_func, X, Y)
        return task.run(seed, random_only=genetic_only, tag=f"{tag}_seed_{seed}")

    start_pool = get_start_pool(config, X, seed)

    reward_func = get_reward_func(config, reward, data, source, task, start_pool, seed)
    task = BayesOptTask(reward_func, acq_func, opt, config)

    top_k = config.get('query_bs', 10)
    num_iter = config.get('num_iter', 200)

    return task.run(start_pool, top_k=top_k, num_iter=num_iter, genetic_only=genetic_only, tag=f"{tag}_seed_{seed}")


source_tasks_dict = {
    # "yap65": [None],
    # "fpbase": ["Stokes Shift"], #, "Quantum Yield"],
    "localfl": [None],
    "stability": [None],
    # "brca1": [None],
    "e4b": [None],
    # "ube2i": [None],
}

def _evaluate(config, acq_func, opt, method):
    reward = config.get("reward", "regex")
    source = config.get("source", None)
    seed = config.get("seed", 0)
    genetic_only = method == "genetic"

    results_cols = [method, reward, seed]
    results = []

    if source is not None:
        for task in source_tasks_dict[source]:
            args = [source, task, acq_func, opt, genetic_only]
            _results = run_trial(config, reward, *args, seed=seed)
            _results = [r + [source] + results_cols for r in _results]
            results += _results
        return results

    for source in source_tasks_dict:
        if source == "fpbase" and reward == "rank":
            continue

        for task in source_tasks_dict[source]:
            args = [source, task, acq_func, opt, genetic_only]
            _results = run_trial(config, reward, *args, seed=seed)
            _results = [r + [source] + results_cols for r in _results]
            results += _results

    return results

def get_surrogate(config):
    config.setdefault('query_bs', 10)
    config.setdefault('num_iter', 200)
    method = config.setdefault("method", "genetic")

    surrogate_config = {
        "model_kwargs": {'p': config.setdefault('dropout_prob', 0.)},
        "ensemble_size": config.setdefault("ensemble_size", config['query_bs']),
        "num_epochs": config.setdefault("num_epochs", 400),
        "bootstrap_ratio": config.setdefault("bootstrap_ratio", None),
        "weight_decay": config.setdefault("weight_decay", 1e-4),
        "holdout_ratio": config.setdefault("holdout_ratio", 0.15),
        "early_stopping": config.setdefault("early_stopping", True),
        "lr": config.setdefault("lr", 1e-3),
        "bs": config.setdefault('bs', 32),
        'patience': config.setdefault('patience', 16),
        "max_shift": config.setdefault("max_shift", 4),
        "mask_size": config.setdefault("mask_size", 2),
    }

    if method == "cnn_deep_ensemble":
        surrogate_config.update(dict(model='CNN'))
        surrogate = surrogates.DeepEnsemble(surrogate_config)
    elif method == "rnn_deep_ensemble":
        surrogate_config.update(dict(model='RNN'))
        surrogate = surrogates.DeepEnsemble(surrogate_config)
    elif method == "embedding_deep_ensemble":
        surrogate_config["finetune"] = config.setdefault("finetune", 'supervised')
        surrogate_config["embedding_type"] = config.setdefault("embedding_type", "BERT")
        surrogate = surrogates.EmbeddingDeepEnsemble(surrogate_config)
    elif method == "embedding_gp" or method == "gp_embedding":
        surrogate_config["finetune"] = config.setdefault("finetune", 'supervised')
        surrogate_config["embedding_type"] = config.setdefault("embedding_type", "BERT")
        surrogate = surrogates.EmbeddingGP(surrogate_config)
    elif method == "ssk_gp":
        surrogate_config["max_len"] = -1 #temporary value
        surrogate = surrogates.SSKGP(surrogate_config)
    elif method == "genetic":
        surrogate = None
    else:
        raise Exception('no valid surrogate type specified')

    if surrogate and torch.cuda.is_available():
        surrogate = surrogate.to("cuda")

    return method, surrogate


def evaluate(config):
    # general settings
    method = config.setdefault("method", "genetic")
    config.setdefault('query_bs', 10)
    config.setdefault('num_iter', 50)

    # construct surrogate
    method, surrogate = get_surrogate(config)

    # construct acquisition
    acq_config = {
        'acq_type': config.setdefault('acq_type', 'ucb'),
        'ucb_beta': config.setdefault('ucb_beta', 0.2)
    }
    if config.get('multi_fidelity', False):
        cost_fn = acquisition.LinearCost(1.)
        noise_model = acquisition.HomoskedasticNoise(config.setdefault('reward_noise_var', 0.1))
        acq_func = acquisition.DiscreteFidelityAcquisition(surrogate, cost_fn, noise_model, config)
    else:
        acq_func = acquisition.SingleFidelityAcquisition(surrogate, acq_config)

    # construct optimizer
    opt_type = config.setdefault("opt_type", "genetic")
    optimizer_kwargs = {
        'alphabet': bo_protein.utils.AMINO_ACIDS,
        'max_len': config.setdefault('max_len', 500),
        'num_evolutions': config.setdefault("num_evolutions", 20),
        'tournament_prob': config.setdefault("tournament_prob", 0.5),
        'population_size': config.setdefault("population_size", 200),
        'p_crossover': config.setdefault("p_crossover", 0.),
        'p_mutation': config.setdefault("p_mutation", 1e-2),
        'weight_pop': config.setdefault('weight_pop', True),
    }
    if opt_type == "genetic":    
        opt = optimizers.GeneticOptimizer(**optimizer_kwargs)
    elif opt_type == "local_genetic":
        opt = optimizers.LocalGeneticOptimizer(max_dist=config.setdefault('max_dist', 5), **optimizer_kwargs)
    elif opt_type == 'thompson':
        opt = optimizers.DiscreteMaxPosteriorSampling(**optimizer_kwargs)
    else:
        raise Exception('no valid optimizer type specified')

    # wandb setup
    try:
        wandb_dir = os.environ["LOGDIR"]
    except KeyError:
        wandb_dir = '.'

    wandb.init(project="bo-protein", config=config, dir=wandb_dir)

    try:
        save_dir = wandb.run.dir
    except Exception as e:
        save_dir = '.'
    save_dir = config.get("save_dir", save_dir)

    results = _evaluate(config, acq_func, opt, method)

    columns = ["num_obj_queries", "candidate_num", "seq", "score", "source", "method", "reward", "seed"]
    additional_cols = list(set(config.keys()) - set(columns))
    columns += additional_cols

    additional_vals = [config[col] for col in additional_cols]
    results = [r + additional_vals for r in results]
    df = pd.DataFrame(results, columns=columns)
    
    fn = os.path.join(save_dir, "results.csv")
    print(f"***** writing to {fn} *****")
    df.to_csv(fn, index=False)

    try:
        wandb.save(fn)
    except:
        pass


def main(**cfg):
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    evaluate(cfg)


if __name__ == "__main__":
    try:
        os.environ["WANDB_DIR"] = os.environ["LOGDIR"]
    except KeyError:
        pass

#     os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", default="online")

    from fire import Fire

    Fire(main)
