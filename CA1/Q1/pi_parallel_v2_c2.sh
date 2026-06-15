#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:10:00
#SBATCH --job-name=pi_v2_c2
#SBATCH --partition=partition

srun --mpi=pmix python pi_parallel_v2.py
