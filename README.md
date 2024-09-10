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

Follow the `notebooks/create_dataset_demo.ipynb` notebook to create stacked training data from model outputs.

## Train the model

### set up a training virtual environment

Start an interactive job requesting GPUs first:
```salloc -p gpu_test -t 0-06:00 --mem=8000 --gres=gpu:1 ```

Then create a new virtual environment using mamba. Change the path after --prefix to your desired location.
```mamba create --prefix /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr_torch python=3.10 pip wheel```

Activate the new conda environment:
```mamba activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr_torch```

Install cuda and pytorch:
```mamba install -c  "nvidia/label/cuda-12.1.0" cuda-toolkit=12.1.0```
```mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia```

Install some other packages:
```mamba install -c conda-forge numpy scipy pandas matplotlib h5py jupyterlab jupyterlab-spellchecker scikit-learn xarray netcdf4 tqdm wandb mlflow```
```pip install nvidia-modulus nvidia-modulus-sym```

