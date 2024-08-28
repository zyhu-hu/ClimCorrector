# Getting Started

## Installation

Here are the steps to install the python virtual environment on Harvard Cannon cluster.

First, create a new virtual environment using mamba. The following command creates a new virtual environment named `climcorr` with python version 3.10. Change the path after --prefix to your desired location.
```
mamba create --prefix /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr python=3.10
```

Next, activate the virtual environment.
```
mamba activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr
```

Then, install the required packages. Run the following command from the root of this repo to install the required packages.

```
pip install .
```

## Create stacked training data from model outputs

Follow the `notebooks/stacked_training_data.ipynb` notebook to create stacked training data from model outputs.