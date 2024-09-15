#!/bin/bash
#SBATCH -A m4331
#SBATCH -C gpu
#SBATCH -q debug 
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 4
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
##SBATCH --mail-user=zeyuanh@nvidia.com
##SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_lstm8th.py --config-name=config \
        climcorr_path='/global/homes/z/zeyuanhu/ClimCorrector/' \
        input_mean='/global/homes/z/zeyuanhu/ClimCorrector/preprocessing/normalization/inputs/input_mean_v1_test1_10year_sub7.npy' \
        input_std='/global/homes/z/zeyuanhu/ClimCorrector/preprocessing/normalization/inputs/input_std_v1_test1_10year_sub7.npy' \
        target_mean='/global/homes/z/zeyuanhu/ClimCorrector/preprocessing/normalization/outputs/target_mean_v1_test1_10year_sub7.npy' \
        target_std='/global/homes/z/zeyuanhu/ClimCorrector/preprocessing/normalization/outputs/target_std_v1_test1_10year_sub7.npy' \
        train_dataset_path='/global/homes/z/zeyuanhu/scratch/climcorr/v1_test1/' \
        val_dataset_path='/global/homes/z/zeyuanhu/scratch/climcorr/v1_test1/' \
        save_path='/global/homes/z/zeyuanhu/scratch/climcorr/saved_models/' \
        expname='lstm_v1_test' \
        batch_size=1024 \
        num_workers=16 \
        epochs=16 \
        loss='huber' \
        save_top_ckpts=15 \
        learning_rate=0.0008 \
        clip_grad=False \
        clip_grad_norm=0.5 \
        logger='wandb' \
        wandb.project='corr_lstm8th_v1' \
        scheduler_name='cosine_warmup' \
        scheduler.cosine_warmup.T_0=16 \
        scheduler.cosine_warmup.eta_min=5.0e-6 "

cd /global/homes/z/zeyuanhu/ClimCorrector/models/lstm_v1/
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"