#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --job-name=pi_v1_c1
#SBATCH --partition=partition
srun --mpi=pmix python pi_parallel_v1.py
