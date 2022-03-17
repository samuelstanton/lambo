# 🏎️🏎️🏎️🏎️
# LaMBO: Accelerating Bayesian Optimization for Protein Design with Denoising Autoencoders

This project uses [Hydra](https://hydra.cc/) for configuration 
and [Weight and Biases](https://docs.wandb.ai/) for logging.

## Installation

#### FoldX
[FoldX](https://foldxsuite.crg.eu/academic-license-info) is available under a free academic license. 
After creating an account you will be emailed a link to download the FoldX executable and supporting assets.
Copy the contents of the downloaded archive to `~/foldx`.
You may also need to rename the FoldX executable (e.g. `mv -v ~/foldx/foldx_20221231 ~/foldx/foldx`).

#### RDKit
[RDKit](https://www.rdkit.org/) is easiest to install if you're using Conda as your package manager (shown below).


```
conda create --name lambo-env python=3.8
conda install -c conda-forge rdkit
pip install -r requirements.txt --upgrade
pip install -e .
```



## Data collection

Example commands:

```bash
python scripts/black_box_opt.py optimizer=mf_genetic task=regex
```

```bash
python scripts/black_box_opt.py surrogate=deep_ensemble acquisition=ehvi optimizer=mb_genetic optimizer/algorithm=soga 
task=proxy_rfp
```

```bash
python scripts/black_box_opt.py surrogate=multi_task_exact_gp acquisition=greedy optimizer=mb_genetic 
optimizer/algorithm=nsga2 task=proxy_rfp optimizer.algorithm.residue_sampler=esm
```

#### Task options
- `regex` (default)
- `chem` (ZINC small molecules, multi-objective)
- `chem` (ZINC small molecules, single-objective)
- `proxy_rfp` (FPBase large molecules, multi-objective)

#### Optimizer options
- `lambo` (default)
- `mb_genetic`
- `mf_genetic`

#### Algorithm options
- `soga` (single-objective)
- `nsga2` (default, multi-objective)

#### Surrogate options
- `deep_ensemble` (default)
- `single_task_exact_gp`
- `multi_task_exact_gp`
- `single_task_svgp`

#### Acquisition options
- `greedy` (default)
- `ehvi`

## Data analysis

See `notebooks/rfp_postprocessing.ipynb` for an example of how to pull down experiment data from
the wandb servers and analyze the results.

## Tests

`pytest tests`

This project currently has very limited test coverage.

## Citation

If you use any part of this code for your own work, please cite

TODO


