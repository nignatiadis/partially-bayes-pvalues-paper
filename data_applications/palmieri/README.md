# Palmieri Microarray Analysis

Applies Neal Algorithm 2 and Neal Algorithm 8 with Pólya trees to the Palmieri microarray dataset.

## Data

- `palmieri_pairwise.csv` - Preprocessed gene expression pairwise differences (input for analysis)
- `palmieri_preprocessing.R` - R script that downloads raw data into `palmieri_data/` and generates `palmieri_pairwise.csv`

Run the preprocessing script first if `palmieri_pairwise.csv` is missing.

## Running the analysis

**On cluster:**

`submit_chain.sh` automates the full workflow:
- Submits `0_burnin.sbatch` which runs `setup_and_burnin.jl`
- Submits 40 sequential batch jobs via `1_batch.sbatch` which runs `run_batch.jl i` for batch index `i` from 1 to 40.

After all jobs complete, run `merge_and_analyze.jl` manually to combine batches and compute final p-values.


## Directories

- `checkpoints/` - MCMC state preserved between batch runs
- `output/` - Individual batch results
- `palmieri_data/` - Raw data files downloaded by preprocessing script
