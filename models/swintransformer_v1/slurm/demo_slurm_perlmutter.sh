#!/bin/bash
#SBATCH -A m4331
#SBATCH -C gpu
#SBATCH -q debug 
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1
#SBATCH -n 1
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
##SBATCH --mail-user=zeyuanh@nvidia.com
##SBATCH --mail-type=ALL
##SBATCH --output=out_%j.out
##SBATCH --error=eo_%j.err

cmd="python train_swintransformer.py --config-name=config \
        climcorr_path='/global/homes/z/zeyuanhu/ClimCorrector/' \
        train_dataset_path='/global/homes/z/zeyuanhu/scratch/climcorr/v1_test1_iter2_processed/' \
        val_dataset_path='/global/homes/z/zeyuanhu/scratch/climcorr/v1_test1_iter2_processed/' \
        save_path='/global/homes/z/zeyuanhu/scratch/climcorr/saved_models/' \
        expname='swin_v1_test' \
        batch_size=16 \
        num_workers=16 \
        epochs=1 \
        early_stop_step=-1 \
        loss='huber' \
        save_top_ckpts=15 \
        learning_rate=0.0008 \
        clip_grad=False \
        clip_grad_norm=0.5 \
        logger='wandb' \
        wandb.project='corr_swin_v1' \
        scheduler_name='cosine_warmup' \
        scheduler.cosine_warmup.T_0=16 \
        scheduler.cosine_warmup.eta_min=5.0e-6 "

cd /global/homes/z/zeyuanhu/ClimCorrector/models/swintransformer_v1/
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"