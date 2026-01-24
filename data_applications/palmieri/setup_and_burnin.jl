DIR = @__DIR__
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Empirikos
using EmpirikosBNP
using CSV
using Statistics   
using MultipleTesting 
using DataFrames
using HypothesisTests
using StatsBase
using Distributions
using Random
using JLD2
using LinearAlgebra

CHECKPOINT_DIR = joinpath(DIR, "checkpoints")
FLIP_SIGNS = any(==("--flip"), ARGS)
FLIP_SEED = let idx = findfirst(==("--flip-seed"), ARGS)
    idx === nothing ? 1 : parse(Int, ARGS[idx + 1])
end
SUFFIX = FLIP_SIGNS ? "_flip" : ""

# Load data
microarray = CSV.File(joinpath(DIR, "palmieri_pairwise.csv")) |> DataFrame
identifiers = names(microarray)[4:end]
diffs = Matrix(microarray[:, identifiers])

# Optional sign-flip null check.
if FLIP_SIGNS
    Random.seed!(FLIP_SEED)
    sign_flips = 2 .* rand(Bernoulli(0.5), size(diffs)) .-1 
    diffs = diffs .* sign_flips
end

# Prepare samples
iidsamples = EmpirikosBNP.IIDSample.(eachrow(diffs))
vars = var.(EmpirikosBNP.iid_samples.(iidsamples)) 
mu_hats = mean.(EmpirikosBNP.iid_samples.(iidsamples))
Ss = Empirikos.ScaledChiSquareSample.(vars, nobs.(iidsamples) .- 1)
config_samples = EmpirikosBNP.ConfigurationSample.(iidsamples)

# T-test p-values
ttests = OneSampleTTest.(eachrow(diffs))
ttest_pvals = pvalue.(ttests)

# Neal2 
Random.seed!(1)
neal2 = EmpirikosBNP.NealAlgorithm2(Ss)
neal2_samples = fit!(neal2; samples=100_000, burnin=5_000)

jldsave(joinpath(CHECKPOINT_DIR, "neal2_results$(SUFFIX).jld2"),
    neal2 = neal2,
    neal2_samples = neal2_samples,
    mu_hats = mu_hats,
    vars = vars,
    Ss = Ss, 
    ttest_pvals = ttest_pvals,
    sign_flips = sign_flips,
    plus_minus = plus_minus,
    flip_seed = FLIP_SEED,
    flip_enabled = FLIP_SIGNS
)

# Setup and burnin for neal8polya
pt = PolyaTreeDistribution(
    base = Empirikos.fold(TDist(8)/std(TDist(8))), 
    J = 8, α = 1.0, 
    ρ = EmpirikosBNP.PTFun(; inner_multiplier=20.0, inner_power=2, 
        boundary_multiplier=0.1, boundary_power=0),
    symmetrized = true,
    median_centered = false
)

neal8polya = EmpirikosBNP.NealAlgorithm8Polya(config_samples; base_polya=pt, neal_cp=deepcopy(neal2))

println("Starting burnin...")
_ = StatsBase.fit!(neal8polya; samples=1, burnin=2_000)


jldsave(joinpath(CHECKPOINT_DIR, "neal8polya_state$(SUFFIX).jld2"), neal8polya = neal8polya)
println("Burnin complete")
