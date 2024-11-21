#!/bin/bash
# submit_all.sh

SLURM_SCRIPT_DIR="./"

for script in ${SLURM_SCRIPT_DIR}*.sh; do
    sbatch $script
done