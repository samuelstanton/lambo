import os
import h5py
import functools
import subprocess
import pandas as pd

rewards = [
	# "regex",
	# "random_nn",
	# "trained_nn",
	"hmm",
	# "embed_dist",
	# "rnafold",
]

NUM_SEEDS = 3

def generate_genetic_results():
	base_dir = "static_genetic"
	if not os.path.exists(base_dir):
		os.mkdir(base_dir)

	script_args = ["python", "scripts/bayes_opt_eval.py"]
	for reward in rewards:
		for seed in range(NUM_SEEDS):
			save_dir = os.path.join(base_dir, f"{reward}_{seed}")
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			reward_args = [f"--reward={reward}",
						   f"--seed={seed}", 
						   f"--save_dir={save_dir}"]
			subprocess.run(script_args + reward_args)

def copy_dataset(f, name, node):
	if not isinstance(node, h5py.Dataset):
		return
		
	f.create_dataset(name, data=node)

def aggregate_results():
	base_dir = "static_genetic"
	if not os.path.exists(base_dir):
		os.mkdir(base_dir)

	combined_fn = os.path.join(base_dir, "results.csv")
	if os.path.exists(combined_fn):
		os.remove(combined_fn)
	dfs = []
	for d in os.listdir(base_dir):
		d = os.path.join(base_dir, d)
		if not os.path.isdir(d):
			continue

		fn = os.path.join(d, "results.csv")
		if not os.path.exists(fn):
			continue

		dfs.append(pd.read_csv(fn))

	df = pd.concat(dfs)
	df.to_csv(combined_fn, index=False)

def main(**cfg):
	generate_genetic_results()
	aggregate_results()

if __name__ == "__main__":
	main()