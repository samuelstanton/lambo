# 🏎️🏎️🏎️🏎️
# LaMBO: [Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders](https://arxiv.org/abs/2203.12742)

## Abstract

Bayesian optimization (BayesOpt) is a gold standard for query-efficient continuous optimization.
However, its adoption for drug and antibody sequence design has been hindered by the discrete, high-dimensional nature of the decision variables. 
We develop a new approach (LaMBO) which jointly trains a denoising autoencoder with a discriminative multi-task Gaussian process head, allowing gradient-based optimization of multi-objective acquisition functions in the latent space of the autoencoder.
These acquisition functions allow LaMBO to balance the explore-exploit tradeoff over multiple design rounds, and to balance objective tradeoffs by optimizing sequences at many different points on the Pareto frontier.
We evaluate LaMBO on a small-molecule task based on the ZINC dataset and introduce a new large-molecule task targeting fluorescent proteins.
In our experiments LaMBO outperforms genetic optimizers and does not require a large pretraining corpus, demonstrating that BayesOpt is practical and effective for biological sequence design.

## Key Results

BayesOpt can be used to maximize the simulated folding stability (-dG) and solvent-accessible surface area (SASA) of red-spectrum fluorescent proteins.
Higher is better for both objectives.
The starting proteins are shown as colored circles, with corresponding optimized offspring shown as crosses.
Stability correlates with protein function (e.g. how long the protein can fluoresce) while SASA is a proxy for fluorescent intensity.

![Figure 1](https://github.com/samuelstanton/lambo/blob/main/lambo/assets/figures/lambo_pareto_front.png?raw=true)

On all three tasks (described in Section 5.1 of the paper), LaMBO outperforms genetic algorithm baselines, specifically NSGA-2 and a model-based genetic optimizer with the same surrogate architecture (MTGP + NEHVI + GA). 
Performance is quantified by the hypervolume bounded by the optimized Pareto frontier.
The midpoint, lower, and upper bounds of each curve depict the 50\%, 20\%, and 80\% quantiles, estimated from 10 trials. 
See Section 5.2 in the paper for more discussion.

![Figure 3](https://github.com/samuelstanton/lambo/blob/main/lambo/assets/figures/lambo_ga_comparison.png?raw=true)


## Installation

#### FoldX
[FoldX](https://foldxsuite.crg.eu/academic-license-info) is available under a free academic license. 
After creating an account you will be emailed a link to download the FoldX executable and supporting assets.
Copy the contents of the downloaded archive to `~/foldx`.
You may also need to rename the FoldX executable (e.g. `mv -v ~/foldx/foldx_20221231 ~/foldx/foldx`).

#### RDKit
[RDKit](https://www.rdkit.org/) is easiest to install if you're using [Conda](https://docs.conda.io/en/latest/)
as your package manager (shown below).


#### TDC
[TDC](https://github.com/mims-harvard/TDC) is required to run the DRD3 docking task.
See the linked README for installation instructions.


```bash
git clone https://github.com/samuelstanton/lambo && cd lambo
conda create --name lambo-env python=3.8 -y && conda activate lambo-env
conda install -c conda-forge rdkit -y
conda install -c conda-forge pytdc pdbfixer openbabel
pip install -r requirements.txt --upgrade
pip install -e .
```

## Reproducing the figures

This project uses [Weight and Biases](https://docs.wandb.ai/) for logging.
The experimental data used to produce the plots in our papers is available [here](https://wandb.ai/samuelstanton/lambo).

See `./notebooks/plot_pareto_front` for a demonstration of how to reproduce Figure 1.

See `./notebooks/plot_hypervolume` for a demonstration of how to reproduce Figures 3 and 4.

## Running the code

See `./notebooks/rfp_preprocessing.ipynb` for a demonstration of how to download PDB files from the [RCSB Protein Data Bank](https://www.rcsb.org/)
and prepare them for use with FoldX.

See `./notebooks/foldx_demo.ipynb` for a demonstration of how to use our Python bindings for FoldX, 
given a starting sequence with known structure.

This project uses [Hydra](https://hydra.cc/) for configuration when running from the command line.

We recommend running NSGA-2 first to test your installation

```bash
python scripts/black_box_opt.py optimizer=mf_genetic optimizer/algorithm=nsga2 task=regex tokenizer=protein
```

For the model-based genetic baseline, run
```bash
python scripts/black_box_opt.py optimizer=mb_genetic optimizer/algorithm=soga optimizer.encoder_obj=mll task=regex tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi
```

For the full LaMBO algorithm, run
```bash
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=regex tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi
```

To evaluate on the multi-objective RFP (large-molecule) or ZINC (small-molecule) tasks,
use `task=proxy_rfp tokenizer=protein` and `task=chem tokenizer=selfies`,
respectively.

To evaluate on the single-objective ZINC task used in papers like
[Tripp et al (2020)](https://proceedings.neurips.cc/paper/2020/hash/81e3225c6ad49623167a4309eb4b2e75-Abstract.html),
run

```bash
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=lanmt task=chem_lsbo tokenizer=selfies surrogate=single_task_svgp acquisition=ei encoder=lanmt_cnn
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
- `single_task_exact_gp` (DKL GP regression)
- `string_kernel_exact_gp` (not recommended, SSK GP regression)
- `deep_ensemble` (MLE regression)

#### Task options
- `regex` (default, maximize counts of 3 bigrams)
- `regex_easy` (maximize counts of 2 tokens)
- `chem` (ZINC small molecules, maximize LogP and QED)
- `chem_lsbo` (ZINC small molecules, maximize penalized LogP)
- `tdc_docking` (ZINC small molecules, minimize DRD3 docking affinity and synthetic accessibility)
- `proxy_rfp` (FPBase large molecules, maximize stability and SASA)

#### Tokenizer options
- `protein` (default, amino acid vocab for large molecules)
- `selfies` (ZINC-derived SELFIES vocab for small molecules)
- `smiles` (not recommended, ZINC-derived SMILES vocab for small molecules)


## Tests

`pytest tests`

This project currently has very limited test coverage.

## Citation

If you use any part of this code for your own work, please cite

```
@misc{stanton2022accelerating,
      title={Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders}, 
      author={Samuel Stanton and Wesley Maddox and Nate Gruver and Phillip Maffettone and Emily Delaney and Peyton Greenside and Andrew Gordon Wilson},
      year={2022},
      eprint={2203.12742},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


