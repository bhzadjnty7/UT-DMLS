#!/bin/bash
#SBATCH --job-name=fedavg_s1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=scenario_1_%j.out
#SBATCH --error=scenario_1_%j.err
#SBATCH --partition=partition

echo "=========================================="
echo "Federated Learning - Scenario 1"
echo "1 Communication Round, 10 Local Epochs"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "Start: $(date)"
echo "=========================================="
echo ""

# Scenario 1: 1 communication round, 10 local epochs
# Total: 4 processes (1 server + 3 clients)
srun --mpi=pmix python logreg_fedavg.py 1 10

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
