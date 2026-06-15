#!/bin/bash
#SBATCH --job-name=fedavg_s2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=scenario_2_%j.out
#SBATCH --error=scenario_2_%j.err
#SBATCH --partition=partition

echo "=========================================="
echo "Federated Learning - Scenario 2"
echo "10 Communication Rounds, 1 Local Epoch"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "Start: $(date)"
echo "=========================================="
echo ""

# Scenario 2: 10 communication rounds, 1 local epoch
# Total: 4 processes (1 server + 3 clients)
srun --mpi=pmix python logreg_fedavg.py 10 1

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
