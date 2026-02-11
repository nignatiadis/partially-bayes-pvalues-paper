#!/bin/bash

#SBATCH --job-name=compound
#SBATCH --array=2-100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10000
#SBATCH --account=pi-ignat
#SBATCH --partition=caslake

julia simulation_median.jl $SLURM_ARRAY_TASK_ID
