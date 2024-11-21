#!/bin/bash
  
#SBATCH -n 1
#SBATCH -o out-1981-%j
#SBATCH -e eo-1981-%j
#SBATCH -p huce_ice
#SBATCH --contiguous
#SBATCH --mail-type=END
#SBATCH --mail-user=zeyuan_hu@fas.harvard.edu
#SBATCH -J create_data_1981 # job name
#SBATCH -t 60
#SBATCH --mem-per-cpu=128000
#SBATCH --no-requeue

cd /n/home00/zeyuanhu/ClimCorrector
source activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr

python preprocessing/preprocess_climcorr_train_data_v2.py 1981
