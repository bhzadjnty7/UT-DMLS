#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --job-name=pi_v2_c1

srun --mpi=pmix python pi_parallel_v2.py
