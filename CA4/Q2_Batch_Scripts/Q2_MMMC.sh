#!/bin/bash

#SBATCH --job-name=Q2_MMMC
#SBATCH --partition=partition
#SBATCH --mem=400mb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/Q2_MMMC_%j.out
#SBATCH --error=logs/Q2_MMMC_%j.err

##### Number of total processes #####
echo "=============================================="
echo "Job: Q2 - Multiple Machines, Multiple Cores (MMMC)"
echo "Using: HuggingFace Accelerate"
echo "=============================================="
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= " $SLURM_NTASKS_PER_NODE
echo "=============================================="

# Activate virtual environment (required as per assignment)
source /home/shared_files/pytorch_venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p checkpoints

# Change this for yourself
export MASTER_PORT=27511
export RENDEZVOUS_ID=$RANDOM

### Get the first node name as master address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"
echo "RENDEZVOUS_ID=$RENDEZVOUS_ID"
echo "=============================================="

# Environment variables
export DATA_DIR=~/data
export CHECKPOINT_DIR=./checkpoints
export OMP_NUM_THREADS=1

echo "Starting training..."

# Multi-node with multiple processes per node
# Use srun to launch on each task, set env vars for distributed
srun --export=ALL bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=4
    export MASTER_ADDR='$MASTER_ADDR'
    export MASTER_PORT='$MASTER_PORT'
    
    python Q2_train.py \
        --epochs 20 \
        --lr 0.001 \
        --batch-size 32 \
        --accumulation-steps 4 \
        --checkpoint-name model_Q2_MMMC.pt
'

echo "=============================================="
echo "Training completed!"
echo "=============================================="
