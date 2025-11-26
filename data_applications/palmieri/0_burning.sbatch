#!/bin/bash
#SBATCH --job-name=palmieri_burnin
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16000
#SBATCH --account=pi-ignat
#SBATCH --partition=caslake
#SBATCH --output=logs/burnin_%j.out

cd "$( dirname "${BASH_SOURCE[0]}" )"
julia setup_and_burnin.jl