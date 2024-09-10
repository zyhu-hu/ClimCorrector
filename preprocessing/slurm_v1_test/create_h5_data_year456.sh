#!/bin/bash
  
#SBATCH -n 1 # TODO number of cpu
#SBATCH -o out-%j
#SBATCH -e eo-%j
#SBATCH -p test
#SBATCH --contiguous
##SBATCH -x holy2a18106
##SBATCH -x holy2b[05101-05108,05201-05208,05301-05308,07101-07108,09202-09208]
#SBATCH --mail-type=END
#SBATCH --mail-user=zeyuan_hu@fas.harvard.edu
#SBATCH -J create_data456 # job name
#SBATCH -t 120                         # job time in minutes
#SBATCH --mem-per-cpu=180000 #MB
#SBATCH --no-requeue


cd /n/home00/zeyuanhu/ClimCorrector
source activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr

python preprocessing/create_h5_data_v1.py \
    'conv_replay_amip_2iter1.cam.h1.1983-*.nc' \
    --data_path '/n/holylfs04/LABS/kuang_lab/Lab/sweidman/cesm_output/conv_replay_amip_2iter1/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/year4/' \
    --start_idx 0 \
    --stride_sample 1 

python preprocessing/create_h5_data_v1.py \
    'conv_replay_amip_2iter1.cam.h1.1984-*.nc' \
    --data_path '/n/holylfs04/LABS/kuang_lab/Lab/sweidman/cesm_output/conv_replay_amip_2iter1/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/year5/' \
    --start_idx 0 \
    --stride_sample 1 

python preprocessing/create_h5_data_v1.py \
    'conv_replay_amip_2iter1.cam.h1.1985-*.nc' \
    --data_path '/n/holylfs04/LABS/kuang_lab/Lab/sweidman/cesm_output/conv_replay_amip_2iter1/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/year6/' \
    --start_idx 0 \
    --stride_sample 1 