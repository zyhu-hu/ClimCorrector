#!/bin/bash
  
#SBATCH -n 1 # TODO number of cpu
#SBATCH -o out
#SBATCH -e eo-%j
#SBATCH -p huce_ice
#SBATCH --contiguous
##SBATCH -x holy2a18106
##SBATCH -x holy2b[05101-05108,05201-05208,05301-05308,07101-07108,09202-09208]
#SBATCH --mail-type=END
#SBATCH --mail-user=zeyuan_hu@fas.harvard.edu
#SBATCH -J rsync # job name
#SBATCH -t 4320                         # job time in minutes
#SBATCH --mem-per-cpu=32000 #MB
#SBATCH --no-requeue


%cd /n/home00/zeyuanhu/ClimCorrector
mamba activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr_torch

python preprocessing/create_h5_data_v1.py \
    'conv_replay_amip_2iter1.cam.h1.1980-*.nc' \
    --data_path '/n/holylfs04/LABS/kuang_lab/Lab/sweidman/cesm_output/conv_replay_amip_2iter1/atm/hist' \
    --save_path '/n/home00/zeyuanhu/holylfs04/preprocessing/test1/' \
    --start_idx 0 \
    --stride_sample 500 