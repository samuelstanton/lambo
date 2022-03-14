#!/usr/bin/env bash

# WANDB credentials
aws s3 cp s3://samuel-stanton-personal-bucket/credentials/.netrc ~/.netrc
# copy source code
aws s3 cp s3://samuel-stanton-personal-bucket/projects/bo-protein/ ./bo-protein/
mv -v ./bo-protein/foldx ~/foldx || echo "foldx dir already exists"
sudo chmod ugo+rwx ~/foldx
sudo chmod ugo+rwx ~/foldx/foldx
cd bo-protein

# venv
export PATH=/opt/venv/bin:$PATH

# install dependencies
python -m pip install -r requirements.txt --upgrade
pip install -e .