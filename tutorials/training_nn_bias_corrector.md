# A Guide for Training Neural Networks to Correct Bias in SPCAM

Author: Zeyuan Hu
02/19/2025

This tutorial demonstrates how to use this repository to train a neural network to correct bias in the Superparameterized Community Atmosphere Model (SPCAM).

## 1. Install the Required Packages for Data Preprocessing

Here are the steps to install the python virtual environment on Harvard Cannon cluster. The purpose of this virtual environment is for data preprocessing and data analysis. We'll have a separate training environment for deep learning.

First, create a new virtual environment using mamba. The following command creates a new virtual environment named `climcorr`. Change the path after --prefix to your desired location. **Please change to your own path**.
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

If you want to add this virtual environment to jupyter notebook, run the following command.
```
python -m ipykernel install --user --name=climcorr
```
After that, you will see that the virtual environment `climcorr` is added to the jupyter notebook and you can select it when you create a new notebook.


## 2. Data Preprocessing

There are two steps for data preprocessing: 1) aggregate the model output nc files to some .h5 files where each contains arrays of input and output data with dimension of (nsamples, nfeatures) and 2) further preprocess the .h5 file to do input and output normalization as well as adding additional attribute information, as well as reshape the nsamples dimension to (ntime, nlat, nlon).

Note that I have preprocessed training data on NCAR's Derecho cluster. You can also directly jump to the next section about the training a PyTorch model on Derecho. The instructions in this section are to illustrate how to reproduce my preprocessing process.

### 2.1 Aggregate the model output nc files to .h5 files

The main tool is the `data_utils` class in the ```utils/data_utils.py``` file. The `data_utils` class is used to aggregate many .nc files and selected input/output variables into single .h5 input and target files. A typical usage of the `data_utils` class can be found in the [../utils/data_utils.py](../utils/data_utils.py).

One tricky thing about the replay data is that each .nc file contains 6 time steps. Step 0 and 3 contains the input at two time steps with an interval of 6 hours. Step 1 and 4 contains the output at those two time steps. Other time steps are not useful. The `data_utils` deals with these time steps selection in the `get_xrdata_input` and `get_xrdata_target` functions.

In the [../utils/data_utils.py](../utils/data_utils.py), I've defined a few version of input/output variables denoted by v1/v2. For example, the `self.v2_inputs_standard` and `self.v2_inputs_attribute` shows all the variables that will be aggregated into the .h5 input file. The `self.v2_outputs` is all the target variables that will be aggregated into the .h5 target file. If you want to add more input features, you can define your own version of input/output variables and add them to the `data_utils` class.

The [../preprocessing/create_h5_data_v2_retrieve_independent.py](../preprocessing/create_h5_data_v2_retrieve_independent.py) script use the `data_utils` class to generate v2 data for the training data. I prepared 8 slurm scripts at [../preprocessing/slurm_v2_test_retrieve_independent/create_h5_data_i.sh](../preprocessing/slurm_v2_test_retrieve_independent/create_h5_data_1.sh) (i=1,2,...,8) to preprocess 40-years SPCAM replay simulation data on the Cannon cluster. To use the slurm script, you need to change the `save_path` to your own path. The slurm script will generate the .h5 files in the `save_path` directory. It will ends up with having 40 folders for each year, and each folder contains a train_input.h5, train_target_dc.h5, train_target_ic.h5, train_target_sum.h5. For the target files, the `sum` file is the total increments that the replay wants to use. The `ic` file is the state-independent component of the increments. The `dc` file is the state-dependent component of the increments. The `input` file is the input features that the replay uses.

The [../preprocessing/slurm_v2_test_retrieve_independent/create_h5_data_4years_sub3_val.sh](../preprocessing/slurm_v2_test_retrieve_independent/create_h5_data_4years_sub3_val.sh) will aggregate the last 4 years of the replay data as the validation data. I subsampled the files by a factor of 3 so that the evaluation during each training epoch can be faster. The validation data will be used to evaluate the model performance during training. During the training, I will also exclude the last 4 years from the training set since it is used as the validation set.

I also have [../preprocessing/slurm_v2_test_retrieve_independent/create_h5_data_40years_sub23.sh](../preprocessing/slurm_v2_test_retrieve_independent/create_h5_data_40years_sub23.sh) to aggregate the entire 40-years replay data. I used this aggregated data to calculate the mean/std of the input and output features. The mean/std will be used to normalize the input and output features during training.

### 2.2 Further preprocess the .h5 file

After aggregating the .nc files to .h5 files, we need to further preprocess the .h5 files to do input and output normalization as well as adding additional attribute information, as well as reshape the nsamples dimension to (ntime, nlat, nlon). This step can potentially be done in your pytorch dataloader if you want to do it on-the-fly. However, doing this preprocessing in advance can save some time during training. 

The [../preprocessing/preprocess_climcorr_train_data_v2.py](../preprocessing/preprocess_v2.py) script is used to preprocess the .h5 files. The script will read the precomputed mean/std of the v2 input and output features in the `../preprocessing/normalization/`.  The script will also add additional attribute information to the .h5 files, such as sine and cosine of longitude, time of day, and time of year. The script will also reshape the nsamples dimension to (ntime, nlat, nlon). I have a script [../preprocessing/generate_slurm_scripts_v2.py](../preprocessing/generate_slurm_scripts_v2.py) to generate 40 slurm scripts under `../preprocessing/slurm_v2_preprocessing/` to preprocess the 40-years replay data. We can then submit the all 40 slurm scripts using [../preprocessing/slurm_v2_preprocessing/submit_all.sh](../preprocessing/slurm_v2_preprocessing/submit_all.sh) to the Cannon cluster to preprocess the data. You should also similarly preprocess the validation data.

After this preprocessing step, you will have a training data folder, which has 40 subfolders for each year. Each subfolder contains the preprocessed .h5 files. You will also have another validation data folder, which should have one subfolder that contains the preprocessed validation data (my pytorch training code assumes training/validation data are always under such subfolders. So if you don't have this subfolder, my training code will get an error of cannot find validation data).

## 3. Training a PyTorch model on NCAR Derecho cluster

