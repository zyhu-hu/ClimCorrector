#!/bin/bash
  
#SBATCH -n 1 # TODO number of cpu
#SBATCH -o out-%j
#SBATCH -e eo-%j
#SBATCH -p huce_ice
#SBATCH --contiguous
##SBATCH -x holy2a18106
##SBATCH -x holy2b[05101-05108,05201-05208,05301-05308,07101-07108,09202-09208]
#SBATCH --mail-type=END
#SBATCH --mail-user=zeyuan_hu@fas.harvard.edu
#SBATCH -J create_dataval # job name
#SBATCH -t 120                         # job time in minutes
#SBATCH --mem-per-cpu=180000 #MB
#SBATCH --no-requeue


cd /n/home00/zeyuanhu/ClimCorrector
source activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr

python preprocessing/create_h5_data_v1_val.py \
    'conv_replay_amip_2iter1.cam.h1.198[8-9]-*.nc' \
    --data_path '/n/holylfs04/LABS/kuang_lab/Lab/sweidman/cesm_output/conv_replay_amip_2iter1/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/' \
    --start_idx 0 \
    --stride_sample 7 

# python preprocessing/create_h5_data_v1.py \
#     'conv_replay_amip_2iter1.cam.h1.1987-*.nc' \
#     --data_path '/n/holylfs04/LABS/kuang_lab/Lab/sweidman/cesm_output/conv_replay_amip_2iter1/atm/hist/' \
#     --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/year8/' \
#     --start_idx 0 \
#     --stride_sample 1 

# python preprocessing/create_h5_data_v1.py \
#     'conv_replay_amip_2iter1.cam.h1.1988-*.nc' \
#     --data_path '/n/holylfs04/LABS/kuang_lab/Lab/sweidman/cesm_output/conv_replay_amip_2iter1/atm/hist/' \
#     --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/year9/' \
#     --start_idx 0 \
#     --stride_sample 1 

# python preprocessing/create_h5_data_v1.py \
#     'conv_replay_amip_2iter1.cam.h1.1989-*.nc' \
#     --data_path '/n/holylfs04/LABS/kuang_lab/Lab/sweidman/cesm_output/conv_replay_amip_2iter1/atm/hist/' \
#     --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/year10/' \
#     --start_idx 0 \
#     --stride_sample 1 