#!/usr/bin/env bash

# WANDB credentials
aws s3 cp ~/.netrc s3://samuel-stanton-personal-bucket/credentials/.netrc

# FoldX source
aws s3 sync ~/foldx/ s3://samuel-stanton-personal-bucket/projects/bo-protein/foldx/

# project source
aws s3 sync ./bo_protein/ s3://samuel-stanton-personal-bucket/projects/bo-protein/bo_protein/
aws s3 sync ./hydra_config/ s3://samuel-stanton-personal-bucket/projects/bo-protein/hydra_config/
aws s3 sync ./scripts/ s3://samuel-stanton-personal-bucket/projects/bo-protein/scripts/
aws s3 cp ./setup.py s3://samuel-stanton-personal-bucket/projects/bo-protein/setup.py
aws s3 cp ./requirements.txt s3://samuel-stanton-personal-bucket/projects/bo-protein/requirements.txt
aws s3 cp ./README.rst s3://samuel-stanton-personal-bucket/projects/bo-protein/README.rst


