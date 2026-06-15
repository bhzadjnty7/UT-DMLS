#!/bin/bash
#SBATCH --job-name=fedavg_mal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=malicious_%j.out
#SBATCH --error=malicious_%j.err
#SBATCH --partition=partition

echo "=========================================="
echo "Federated Learning - Malicious Client"
echo "10 Rounds, 1 Epoch, Client 2 LR=0.5"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="
echo ""

# Malicious client experiment
# 10 rounds, 1 local epoch, Client 2 has LR=0.5
srun --mpi=pmix python malicious.py

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
