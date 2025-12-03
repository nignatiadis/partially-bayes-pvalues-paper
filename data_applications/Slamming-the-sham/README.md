# Slamming-the-Sham Chicken Experiment

Applies partially Bayes p-values to the chicken experiment comparing sham vs exposed groups.

## Data

- `chickens.dat` - Experiment data with frequency, sham, and exposed group measurements

## Running the analysis

Run the R script:
```R
Rscript chickens.R
```

This fits the Stan model (`partially_bayes.stan`) to the data and computes partially Bayes p-values. Outputs two plots:
- `bs_boxplots.pdf` - Posterior distributions of bias parameters
- `pvalue_boxplots.pdf` - P-value comparisons (partially Bayes, exposed-only, sham)

## Files

- `chickens.R` - Main analysis script
- `partially_bayes.stan` - Hierarchical Stan model for nuisance parameters
- `chickens.dat` - Input data
