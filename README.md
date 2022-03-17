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


## Running the code

Example commands:

```bash
python scripts/black_box_opt.py optimizer=mf_genetic optimizer/algorithm=nsga2 task=regex 
```

```bash
python scripts/black_box_opt.py surrogate=deep_ensemble acquisition=ehvi optimizer=mb_genetic optimizer/algorithm=soga 
task=proxy_rfp
```

```bash
python scripts/black_box_opt.py surrogate=multi_task_exact_gp acquisition=greedy optimizer=mb_genetic 
optimizer/algorithm=nsga2 task=proxy_rfp optimizer.algorithm.residue_sampler=esm
```

Below we list significant configuration options.
See the config files in `./hydra_config` for all configurable parameters.
Note that any config field can be overridden from the command line, and some configurations are not supported. 

#### Acquisition options
- `nehvi` (default, multi-objective)
- `ehvi` (multi-objective)
- `ei` (single-objective)
- `greedy` (single and multi-objective)

#### Encoder options
- `mlm_cnn` (default, substitutions only)
- `mlm_transformer` (substitutions only)
- `lanmt_cnn` (substitutions, insertions, deletions)
- `lanmt_transformer` (substitutions, insertions, deletions)

#### Optimizer options
- `lambo` (default)
- `mb_genetic` (Genetic baseline with model-based compound screening)
- `mf_genetic` (Model-free genetic baseline)

#### Algorithm options
- `soga` (default, single-objective)
- `nsga2` (multi-objective)

#### Surrogate options
- `multi_task_exact_gp` (default, DKL MTGP regression)
- `single_task_svgp` (DKL SVGP regression)
- `string_kernel_exact_gp` (SSK GP regression)
- `single_task_exact_gp` (DKL GP regression)
- `deep_ensemble` (MLE regression)

#### Task options
- `regex` (default, maximize counts of 3 bigrams)
- `regex_easy` (maximize counts of 2 tokens)
- `chem` (ZINC small molecules, maximize LogP and QED)
- `chem_lsbo` (ZINC small molecules, maximize penalized LogP)
- `proxy_rfp` (FPBase large molecules, maximize stability and SASA)

#### Tokenizer options
- `protein` (default, amino acid vocab for large molecules)
- `selfies` (ZINC-derived SELFIES vocab for small molecules)
- `smiles` (not recommended, ZINC-derived SMILES vocab for small molecules)


## Data analysis

See `notebooks/rfp_postprocessing.ipynb` for an example of how to pull down experiment data from
the wandb servers and analyze the results.

## Tests

`pytest tests`

This project currently has very limited test coverage.

## Citation

If you use any part of this code for your own work, please cite

TODO


