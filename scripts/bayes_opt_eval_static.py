import os
import sys
import h5py
import wandb
import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch

from regression import ind_splits
from bo_protein import rewards
from scripts.bayes_opt_eval import get_surrogate, \
								   get_reward_func, \
								   source_tasks_dict

def print_hdf5(fn):
	with h5py.File(fn, "r") as fd:
		fd.visititems(lambda name, node: print(name))	

def load_trial(reward, source, task, seed):
	fn = 'static_genetic/results.csv'
	df = pd.read_csv(fn)
	df = df.loc[(df["reward"] == reward) &
	 	   		(df["source"] == source) &
	 	   		(df["seed"] == seed)]
	
	print(df)

	df = df[["seq", "score"]].to_numpy()
	# idx = df.groupby(["num_obj_queries"])["score"].transform(max) == df["score"]
	# df = df[idx].drop_duplicates(["num_obj_queries","score"])

	print(df)

	proposals, scores = df[:,0], df[:,1].astype(np.float32)
	return proposals, scores

def test_distribution_shift(config, reward, source, task, seed):
	proposals, scores = load_trial(reward, source, task, seed)

	task = task if task else ""
	tag = f'{source}_{task.lower().replace(" ", "_")}'

	num_iter = len(proposals)
	num_train = num_iter // 2

	X_train = proposals[:num_train]
	Y_train = scores[:num_train]
	X_test = proposals[num_train:]
	Y_test = scores[num_train:]

	np.random.seed(seed)
	train_idx = np.arange(num_train)
	test_idx = np.arange(num_iter - num_train)
	np.random.shuffle(train_idx)
	np.random.shuffle(test_idx)

	num_examples = config.get("num_examples", 250)
	X_train = X_train[train_idx][:num_examples]
	Y_train = Y_train[train_idx][:num_examples]
	X_test = X_test[test_idx][:num_examples]
	Y_test = Y_test[test_idx][:num_examples]

	Y_mean, Y_std = np.mean(Y_train), np.std(Y_train)
	Y_train = (Y_train - Y_mean) / np.where(Y_std != 0, Y_std, 1)
	Y_test = (Y_test - Y_mean) / np.where(Y_std != 0, Y_std, 1)

	_, surrogate = get_surrogate(config)

	X = np.concatenate([X_train, X_test], axis=0)
	max_len = max([len(x) for x in X])
	if surrogate and hasattr(surrogate, 'max_len'):
		surrogate.max_len = max_len 

	surrogate.fit(
		X_train,
		Y_train,
		X_test,
		Y_test,
		log_prefix=f"train_{tag}_distribution_shift",
	)

	bs = config.get("bs", 200)
	metrics = surrogate.evaluate(X_test, Y_test, bs=bs, 
								 log_prefix=f"eval_{tag}_distribution_shift")

	print("***** regression on distribution shift metrics *****")
	pprint.pprint(metrics)
	print("\n")

def test_reward_regression(config, reward, source, task, seed):
	data, max_len, tag = ind_splits(source, task, seed=seed)
	X = np.concatenate([data[0], data[2]])

	num_examples = config.get("num_examples", 250)
	X = X[:num_examples]

	reward_func = get_reward_func(config, reward, data, source, task, None, seed)
	Y = reward_func.score(X)
	Y = Y.cpu().numpy() if torch.is_tensor(Y) else Y


	Y_mean, Y_std = np.mean(Y), np.std(Y)
	Y = (Y - Y_mean) / np.where(Y_std != 0, Y_std, 1)

	X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y, test_size=0.1
	)

	_, surrogate = get_surrogate(config)

	if surrogate and hasattr(surrogate, 'max_len'):
		surrogate.max_len = max_len 

	surrogate.fit(
		X_train,
		Y_train,
		X_test,
		Y_test,
		log_prefix=f"train_{tag}_reward_regression",
	)

	bs = config.get("bs", 200)
	metrics = surrogate.evaluate(X_test, Y_test, bs=bs, 
								 log_prefix=f"eval_{tag}_reward_regression")

	print("***** regression on reward metrics *****")
	pprint.pprint(metrics)
	print("\n")

def test_target_regression(config, source, task, seed):
	split = 0.9
	data, max_len, tag = ind_splits(source, task, split=split, seed=seed)
	X_train, Y_train, X_test, Y_test = data 

	num_examples = config.get("num_examples", 250)
	num_train = int(num_examples * split)
	num_test = int(num_examples * (1 - split))
	X_train = X_train[:num_train]
	Y_train = Y_train[:num_train]
	X_test = X_test[:num_test]
	Y_test = Y_test[:num_test]

	Y = np.concatenate([Y_train, Y_test], axis=0)
	Y_mean, Y_std = np.mean(Y), np.std(Y)
	Y_train = (Y_train - Y_mean) / np.where(Y_std != 0, Y_std, 1)
	Y_test = (Y_test - Y_mean) / np.where(Y_std != 0, Y_std, 1)

	_, surrogate = get_surrogate(config)

	if surrogate and hasattr(surrogate, 'max_len'):
		surrogate.max_len = max_len 

	surrogate.fit(
		X_train,
		Y_train,
		X_test,
		Y_test,
		log_prefix=f"train_{tag}_target_regression",
	)

	bs = config.get("bs", 200)
	metrics = surrogate.evaluate(X_test, Y_test, bs=bs, 
								 log_prefix=f"eval_{tag}_target_regression")

	print("***** regression on target metrics *****")
	pprint.pprint(metrics)
	print("\n")

def run_tests(config):
	method = config.get("method", "genetic")
	reward = config.get("reward", "regex")
	source = config.get("source", "localfl")
	task = config.get("task", source_tasks_dict[source][0])
	seed = config.get("seed", 0)

	num_examples = config.get("num_examples", 250)
	if method == "ssk_gp" and source == "localfl" and num_examples > 250:
		return

	if not "reward" in config:
		test_target_regression(config, source, task, seed)
	else:
		test_reward_regression(config, reward, source, task, seed)
		test_distribution_shift(config, reward, source, task, seed)

def main(**cfg):
	try:
		wandb_dir = os.environ["LOGDIR"]
	except KeyError:
		wandb_dir = '.'

	wandb.init(project="bo-protein", config=cfg, dir=wandb_dir)

	if torch.cuda.is_available():
		torch.set_default_tensor_type(torch.cuda.FloatTensor)

	run_tests(cfg)

if __name__ == "__main__":
	try:
		os.environ["WANDB_DIR"] = os.environ["LOGDIR"]
	except KeyError:
		pass

	os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", default="dryrun")

	from fire import Fire

	Fire(main)
