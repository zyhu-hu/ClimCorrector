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
#SBATCH -J create_data # job name
#SBATCH -t 600                         # job time in minutes
#SBATCH --mem-per-cpu=180000 #MB
#SBATCH --no-requeue


cd /n/home00/zeyuanhu/ClimCorrector
source activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr

python preprocessing/create_h5_data_v2_retrieve_independent.py \
    'spcam_replay_2iter4.cam.h1.2010-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2/2010/' \
    --start_idx 0 \
    --stride_sample 1 

python preprocessing/create_h5_data_v2_retrieve_independent.py \
    'spcam_replay_2iter4.cam.h1.2011-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2/2011/' \
    --start_idx 0 \
    --stride_sample 1 


python preprocessing/create_h5_data_v2_retrieve_independent.py \
    'spcam_replay_2iter4.cam.h1.2012-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2/2012/' \
    --start_idx 0 \
    --stride_sample 1 


python preprocessing/create_h5_data_v2_retrieve_independent.py \
    'spcam_replay_2iter4.cam.h1.2013-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2/2013/' \
    --start_idx 0 \
    --stride_sample 1 


python preprocessing/create_h5_data_v2_retrieve_independent.py \
    'spcam_replay_2iter4.cam.h1.2014-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2/2014/' \
    --start_idx 0 \
    --stride_sample 1 

