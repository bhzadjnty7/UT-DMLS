#!/bin/bash

#SBATCH --job-name=Q2_SMSC
#SBATCH --partition=partition
#SBATCH --mem=400mb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/Q2_SMSC_%j.out
#SBATCH --error=logs/Q2_SMSC_%j.err

##### Number of total processes #####
echo "=============================================="
echo "Job: Q2 - Single Machine, Single Core (SMSC)"
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

# Environment variables
export DATA_DIR=~/data
export CHECKPOINT_DIR=./checkpoints
export OMP_NUM_THREADS=1

echo "Starting training..."
echo "=============================================="

# Single node - accelerate launch directly (no srun needed for single node)
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    Q2_train.py \
    --epochs 20 \
    --lr 0.001 \
    --batch-size 32 \
    --accumulation-steps 4 \
    --checkpoint-name model_Q2_SMSC.pt

echo "=============================================="
echo "Training completed!"
echo "=============================================="
