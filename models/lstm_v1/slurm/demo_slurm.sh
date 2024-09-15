#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node 1
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mail-user=zeyuan_hu@fas.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err
#SBATCH --constraint="[a100|v100|h100]"
#SBATCH --exclude=holygpu8a[25104,22505,22405,26304]

NUM_TASKS=1
NUM_CPUS=16


cmd="python train_lstm8th.py --config-name=config \
        data_path='/n/holyscratch01/kuang_lab/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/' \
        expname='/n/holyscratch01/kuang_lab/zeyuanhu/scratch/climcorr_preprocessing/v1_test1/' \
        batch_size=1024 \
        num_workers=8 \
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

cd /n/home00/zeyuanhu/ClimCorrector/models/lstm_v1
module load Mambaforge/22.11.1-fasrc01
source activate /n/holylfs04/LABS/kuang_lab/Lab/kuanglfs/zeyuanhu/mamba_env/climcorr_torch
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | awk '/MIG/ {gsub(/[()]/,"");print $NF}')
srun  --ntasks=$NUM_TASKS --cpus-per-task=$NUM_CPUS --gres=gpu:$NUM_TASKS bash -c "source ddp_export.sh && $cmd"