#!/bin/bash
#PBS -N swin_transformer
#PBS -A UHAR0026
#PBS -j oe
#PBS -k eod
#PBS -q preempt 
#PBS -r y
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=128GB
#PBS -l gpu_type=a100
#PBS -l job_priority=economy

# Load required modules
module load apptainer

# Define distributed environment variables
NUM_GPUS_PER_NODE=4
NUM_NODES=$(wc -l < $PBS_NODEFILE)  # Number of nodes
export WORLD_SIZE=$(($NUM_GPUS_PER_NODE * $NUM_NODES))  # Total GPUs
export NODE_RANK=$(cat $PBS_NODEFILE | nl | grep $(hostname) | awk '{print $1 - 1}')  # Node rank (0-indexed)
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)  # Master node hostname
export MASTER_PORT=29500  # Default port

# Debugging
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Define container and script variables
CONTAINER_PATH="/glade/campaign/univ/uhar0026/zeyuanhu/tutorial/apptainer/my_modulus.sif"
WORK_DIR="/glade/campaign/univ/uhar0026/zeyuanhu/tutorial/ClimCorrector/models/swintransformer_v2"
CMD="python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --use_env train_swintransformer.py --config-name=config \
        climcorr_path='/glade/campaign/univ/uhar0026/zeyuanhu/tutorial/ClimCorrector/' \
        train_dataset_path='/glade/campaign/univ/uhar0026/zeyuanhu/v3_iter2_processed/' \
        val_dataset_path='/glade/campaign/univ/uhar0026/zeyuanhu/v3_iter2_4year_sub3_processed/' \
        target_filename='target_sum' \
        variable_subsets='v2' \
        save_path='/glade/campaign/univ/uhar0026/zeyuanhu/tutorial/saved_models/' \
        expname='swinv3_dim1024_depth8_soap_tutorial' \
        batch_size=4 \
        num_workers=16 \
        epochs=2 \
        early_stop_step=-1 \
        loss='huber' \
	optimizer='soap' \
	restart_full_ckpt=True \
        save_top_ckpts=15 \
        learning_rate=0.003 \
        clip_grad=False \
        clip_grad_norm=0.5 \
	swin.in_chans=142 \
        swin.window_size_x=4 \
        swin.window_size_y=6 \
        swin.embed_dim=128 \
        swin.depths=4 \
        swin.num_heads=8 \
        swin.mlp_ratio=2.0 \
        swin.checkpoint_stages=False \
        logger='wandb' \
	wandb.project='tutorial' \
        scheduler_name='cosine_warmup' \
        scheduler.cosine_warmup.T_0=40 \
        scheduler.cosine_warmup.eta_min=1.0e-5"

# Change to working directory
cd $WORK_DIR

# Run inside the container
singularity exec \
    --nv --cleanenv \
    --bind /glade/work \
    --bind /glade/campaign \
    $CONTAINER_PATH \
    bash -c "$CMD"