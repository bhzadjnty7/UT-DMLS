#!/bin/bash

#SBATCH --job-name=Q2_SMMC
#SBATCH --partition=partition
#SBATCH --mem=400mb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2   
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/Q2_SMMC_%j.out
#SBATCH --error=logs/Q2_SMMC_%j.err

source /home/shared_files/pytorch_venv/bin/activate

mkdir -p logs checkpoints
export DATA_DIR=~/data
export CHECKPOINT_DIR=./checkpoints
export OMP_NUM_THREADS=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Starting training (Corrected SMMC)..."

srun --export=ALL bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=2
    export MASTER_ADDR='$MASTER_ADDR'
    export MASTER_PORT='$MASTER_PORT'
    
    python Q2_train.py \
        --epochs 20 \
        --lr 0.001 \
        --batch-size 32 \
        --accumulation-steps 4 \
        --checkpoint-name model_Q2_SMMC.pt
'
