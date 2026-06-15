#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:10:00
#SBATCH --job-name=pi_v1_c2

srun --mpi=pmix python pi_parallel_v1.py
