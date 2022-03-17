from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, "README.md")) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

#TODO: read this in properly
install_requires = [
    # "git+https://github.com/cornellius-gp/gpytorch.git",
    # "git+https://github.com/pytorch/botorch.git",
    # "git+https://github.com/henrymoss/BOSS.git",
    # "tape_proteins",
    # "emukit",
    # "python-levenshtein",
    # "wandb==0.10.28",
    # "fair-esm==0.3.1",
    # "fire==0.4.0",
    # "xgboost==1.4.1",
    # "hmmlearn==0.2.5",
    # "h5py==3.2.1",
    # "biopython>=1.79",
]

desc = "Code to reproduce experiments from Accelerating Bayesian Optimization for Protein Design" \
       "with Denoising Autoencoders"

setup(
    name="lambo",
    version="0.1.3",
    description=desc,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Samuel Stanton, Wesley Maddox, and Nate Gruver",
    author_email="ss13641@nyu.edu",
    url="https://github.com/samuelstanton/lambo.git",
    license="Apache-2.0",
    packages=["bo_protein"],
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
    ],
)