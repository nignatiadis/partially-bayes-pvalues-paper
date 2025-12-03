# Partially Bayes p-values for large scale inference

Reproduction code for the working paper "Partially Bayes p-values for large scale inference"

## Authors

- Nikolaos Ignatiadis (ignat@uchicago.edu)
- Li Ma (li.ma@uchicago.edu)

## Abstract

We seek to conduct statistical inference for a large collection of primary parameters, each with its own nuisance parameters. Our approach is partially Bayesian, in that we treat the primary parameters as fixed while we model the nuisance parameters as random and drawn from an unknown distribution which we endow with a nonparametric prior. We compute partially Bayes p-values by conditioning on nuisance parameter statistics, that is, statistics that are ancillary for the primary parameters and informative about the nuisance parameters. The proposed p-values have a Bayesian interpretation as tail areas computed with respect to the posterior distribution of the nuisance parameters. Similarly to the conditional predictive p-values of Bayarri and Berger, the partially Bayes p-values avoid double use of the data (unlike posterior predictive p-values). A key ingredient of our approach is that we model nuisance parameters hierarchically across problems; the sharing of information across problems leads to improved calibration.

We illustrate the proposed partially Bayes p-values in two applications: the normal means problem with unknown variances and a location-scale model with unknown distribution shape. We model the scales via Dirichlet processes in both examples and the distribution shape via Pólya trees in the second. Our proposed partially Bayes p-values increase power and calibration compared to purely frequentist alternatives.

**Keywords:** Nuisance parameters, Bayesian nonparametrics, compound p-values, Dirichlet process mixture models, Pólya trees

## Setup

This project uses Julia with dependencies managed through `Project.toml` and `Manifest.toml`. The `Project.toml` lists the direct dependencies, while `Manifest.toml` locks the exact versions of all packages (including transitive dependencies) to ensure reproducibility.

To set up the environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()  # installs exact versions from Manifest.toml
```

Key dependencies include `Empirikos`, `EmpirikosBNP`, `Distributions`, `JLD2`, `CSV`, `DataFrames`, `PGFPlotsX`, and `MultipleTesting`.

## Reproducing the Simulations

The simulations compare four methods: standard t-tests, Neal Algorithm 2 (normal base), Neal Algorithm 8 with Pólya trees, and an oracle that knows the true data-generating distribution.

### Running simulations

The simulation script takes a task ID (1-1000) that determines the parameter combination:
- 2 variance distributions (Dirac, Uniform)
- 5 Subbotin parameters (1.0, 1.5, 2.0, 2.5, 3.0)
- 100 seeds

**To test locally:**
```bash
julia simulation.jl 1
```

**For full results (on SLURM cluster):**
```bash
sbatch simulations.sbatch
```

This submits an array job with 1000 tasks. Each task saves results to `simulation_results/method_res_{task_id}.jld2`.

### Analyzing results

After simulations complete:

```bash
# Aggregate all results into a CSV
julia analyze_simulations.jl

# Generate plots
julia plot_simulations.jl
julia plot_simulation_polya.jl
```

The `analyze_simulations.jl` script loads all JLD2 files, computes means/standard deviations grouped by method and parameters, and outputs `aggregated_results.csv`.

## Data Applications

### Palmieri microarray analysis

See `data_applications/palmieri/README.md` for instructions. 

### Slamming-the-sham chicken experiment

See `data_applications/Slamming-the-sham/README.md` for instructions.