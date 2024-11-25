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
#SBATCH -J create_data2 # job name
#SBATCH -t 600                         # job time in minutes
#SBATCH --mem-per-cpu=180000 #MB
#SBATCH --no-requeue


cd /n/home00/zeyuanhu/ClimCorrector
source activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr

python preprocessing/create_h5_data_v2_retrieve_independent_nosubsampling.py \
    'spcam_replay_2iter3.cam.h1.2005-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2_include_lastfile/2005/' \



python preprocessing/create_h5_data_v2_retrieve_independent_nosubsampling.py \
    'spcam_replay_2iter3.cam.h1.2006-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2_include_lastfile/2006/' \




python preprocessing/create_h5_data_v2_retrieve_independent_nosubsampling.py \
    'spcam_replay_2iter3.cam.h1.2007-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2_include_lastfile/2007/' \




python preprocessing/create_h5_data_v2_retrieve_independent_nosubsampling.py \
    'spcam_replay_2iter3.cam.h1.2008-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2_include_lastfile/2008/' \




python preprocessing/create_h5_data_v2_retrieve_independent_nosubsampling.py \
    'spcam_replay_2iter3.cam.h1.2009-*.nc' \
    --data_path '/n/home04/sweidman/holylfs04/CESM215_out/Run/archive/spcam_replay_2iter/atm/hist/' \
    --save_path '/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2_include_lastfile/2009/' \



