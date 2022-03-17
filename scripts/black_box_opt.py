import hydra
import wandb
import warnings
import random
import logging
import os
from pathlib import Path
import sys
import torch

from omegaconf import OmegaConf

from upcycle.scripting import startup
from upcycle.logging.analysis import flatten_config

from gpytorch.settings import max_cholesky_size


@hydra.main(config_path='../hydra_config', config_name='black_box_opt')
def main(config):
    # setup
    random.seed(None)  # make sure random seed resets between multirun jobs for random job-name generation
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    wandb.init(project='lambo', config=log_config, mode=config.wandb_mode,
               group=config.exp_name)
    config['job_name'] = wandb.run.name
    config, _ = startup(config)  # random seed is fixed here

    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # catch really annoying BioPython warnings

        try:
            # create initial candidates, dataset
            tokenizer = hydra.utils.instantiate(config.tokenizer)
            bb_task = hydra.utils.instantiate(config.task, tokenizer=tokenizer, candidate_pool=[])

            project_root = Path(os.getcwd()).parents[2]  # changing the Hydra run dir will break this.
            base_candidates, base_targets, all_seqs, all_targets = bb_task.task_setup(config, project_root=project_root)

            # optimizer
            max_chol_sz = config.surrogate.get('max_cholesky_size', int(1e5))
            with max_cholesky_size(max_chol_sz):
                optimizer = hydra.utils.instantiate(
                    config.optimizer,
                    bb_task=config.task,
                    surrogate=config.surrogate,
                    acquisition=config.acquisition,
                    encoder=config.encoder,
                    tokenizer=tokenizer
                )
                metrics = optimizer.optimize(
                    base_candidates, base_targets, all_seqs, all_targets, log_prefix=config.task.log_prefix
                )
            metrics = {key.split('/')[-1]: val for key, val in metrics.items()}  # strip prefix
            ret_val = metrics['hypervol_rel']

        except Exception as err:
            logging.exception(err)
            ret_val = float('NaN')

    wandb.finish()  # necessary to log Hydra multirun output to different jobs
    return ret_val


if __name__ == "__main__":
    main()
    sys.exit()
