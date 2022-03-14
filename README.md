# 🏎️🏎️🏎️🏎️
# LaMBO: Accelerating Bayesian Optimization for Protein Design with Denoising Autoencoders

This project uses [Hydra](https://hydra.cc/) for configuration 
and [Weight and Biases](https://docs.wandb.ai/) for logging.

## Installation

```
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
- `proxy_rfp`

#### Optimizer options
- `mb_genetic` (default)
- `mf_genetic`

#### Algorithm options
- `nsga2` (default, multi-objective)
- `soga` (single-objective)
- `isres` (single-objective)

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
