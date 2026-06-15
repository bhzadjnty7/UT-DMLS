#!/bin/bash

#SBATCH --job-name=Q1_MMMC_NoAccum
#SBATCH --partition=partition
#SBATCH --mem=400mb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/Q1_MMMC_NoAccum_%j.out
#SBATCH --error=logs/Q1_MMMC_NoAccum_%j.err

##### Print job information #####
echo "=============================================="
echo "Job: MMMC WITHOUT Gradient Accumulation"
echo "=============================================="
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= " $SLURM_NTASKS_PER_NODE
echo "=============================================="

# Activate virtual environment
source /home/shared_files/pytorch_venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p checkpoints

# Change this for yourself (use unique port)
export MASTER_PORT=27424
export RENDEZVOUS_ID=$RANDOM
export WORLD_SIZE=4

### Get the first node name as master address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"
echo "=============================================="

# Suppress threading warning
export OMP_NUM_THREADS=1

# Set data directory
export DATA_DIR=~/data
export CHECKPOINT_DIR=./checkpoints

# Run training WITHOUT gradient accumulation
echo "Starting training (NO gradient accumulation)..."
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_id=$RENDEZVOUS_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    Q1_train.py \
    --epochs 20 \
    --lr 0.001 \
    --batch-size 128 \
    --no-accumulation \
    --checkpoint-name model_MMMC_no_accum.pt

echo "=============================================="
echo "Training completed!"
echo "=============================================="
