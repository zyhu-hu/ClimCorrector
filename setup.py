#!/usr/bin/env python

import os
from setuptools import find_packages, setup

# Check environment variable for the backend choice
ml_backend = os.getenv('ML_BACKEND', 'pytorch').lower()

# Base requirements
install_requires = [
    "xarray",
    "numpy",
    "pandas",
    "matplotlib",
    "netCDF4",
    "h5py",
    "tqdm",
]

# Conditional requirements based on the backend choice
if ml_backend == 'pytorch':
    install_requires.append('torch')
elif ml_backend == 'tensorflow':
    install_requires.append('tensorflow')
else:
    raise ValueError(f"Unsupported ML_BACKEND value: {ml_backend}. Choose 'tensorflow' or 'pytorch'.")

setup(
    name="climcorrector_utils",
    version="0.0.1",
    description="""
    Tools for working with ClimCorrector.
    """,
    author="Zeyuan Hu",
    author_email="zeyuan_hu@fas.harvard.edu",
    python_requires=">=3.10",
    install_requires=install_requires,
    packages=find_packages(),
)
