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

desc = "Code to reproduce experiments from Accelerating Bayesian Optimization for Protein Design" \
       "with Denoising Autoencoders."

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
    packages=["lambo"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
    ],
)