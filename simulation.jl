task_id = parse(Int, ARGS[1])  # SLURM_ARRAY_TASK_ID

using Pkg
Pkg.activate(".")

using MultipleTesting
using EmpirikosBNP
using Empirikos
using Distributions
using HypothesisTests
using StatsBase
using Random
using JLD2
using RCall
using MosekTools

dir = @__DIR__

# Overal simulation strategy

# Uniform, Normal, Laplace
# K=12 (for now), 
# All variances the same, or variances two point prior
# 10% alternatives?
# Report avg number null p-values <= 0.01 & power
# FDR of BH at 0.1 & power_BH
# Repeat: 50 times for each setting?
# Store results objects, p-values, power results.
# Both t and Normal base measures.

n = 10_000
n1 = 1_000 #alternative proportion
K = 12
α_bh = 0.1
α_pval = 0.01

variance_distributions = (
    Dirac = Dirac(1.0),
    Uniform = Uniform(0.5, 2.0)
)

subbotin_parameters = (1.0, 1.5, 2.0, 2.5, 3.0)


seeds = 1:100

key_combinations = collect(Iterators.product(variance_distributions, subbotin_parameters, seeds))

variance_dbn, subbotin_param, seed = key_combinations[task_id]
effect_size = 2.5

dbn = PGeneralizedGaussian(0, 1, subbotin_param) / std(PGeneralizedGaussian(0, 1, subbotin_param))

Random.seed!(1)
Zs_mat_init =  rand(dbn, n, K)
Hs = sample(1:n,n1, replace=false)
σs_squared = rand(variance_dbn, n)
for i in Base.OneTo(seed)
    Zs_mat_init .= rand(dbn, n, K)
    Hs .= sample(1:n,n1, replace=false)
    σs_squared .= rand(variance_dbn, n)
end
σs = sqrt.(σs_squared)
Hs_bool = falses(n)
Hs_bool[Hs] .= true
μs = zeros(n)
μs[Hs_bool] .= effect_size ./ sqrt(K)
Zs_mat = (Zs_mat_init .+ μs) .* σs

iidsamples = EmpirikosBNP.IIDSample.(collect.(eachrow(Zs_mat)))
mu_hats = mean.(getfield.(iidsamples, :Z))
vars = var.(EmpirikosBNP.iid_samples.(iidsamples))



# evaluation if only have indicator of discoveries
function evaluate_rjs(Hs, rj_idx)
    discoveries = sum(rj_idx)
    true_discoveries = sum(rj_idx .& Hs)
    false_discoveries = discoveries - true_discoveries
    FDP = false_discoveries / max(discoveries, 1)
    Power = true_discoveries / max( sum(Hs), 1)

    (   
    FDP_BH = FDP, 
    Power_BH = Power, 
    discoveries_BH = discoveries
    )
end

# evaluation with p-values
function evaluate_pvals(Hs, pvals; α_pval = α_pval, α_bh = α_bh)

    uniformity_pval = mean(pvals[.! Hs_bool] .<= α_pval) 
    power_pval = mean(pvals[Hs_bool] .<= α_pval)
    discoveries_pval = sum(pvals .<= α_pval)

    bh_pvals = MultipleTesting.adjust(pvals, BenjaminiHochberg())
    rj_idx = bh_pvals .<= α_bh
    bh_results = evaluate_rjs(Hs, rj_idx)
    (
        Uniformity_Pval = uniformity_pval,
        Power_Pval = power_pval,
        discoveries_Pval = discoveries_pval,
        bh_results...
    )
end



# SENS 

sens_file = joinpath(dir, "SENS", "SENS_standalone.R")
R"""
source($(sens_file))
"""


R"""
FDR_res_general <- SENS($Zs_mat, $α_bh, option='General')
FDR_res_gaussian <- SENS($Zs_mat, $α_bh, option='Gaussian')
"""
R"""
SENS_rjs_general = FDR_res_general$de
SENS_rjs_gaussian = FDR_res_gaussian$de
"""
@rget SENS_rjs_general
@rget SENS_rjs_gaussian


ttests = OneSampleTTest.(response.(iidsamples))
ttest_pvals = pvalue.(ttests)

Ss = Empirikos.ScaledChiSquareSample.(vars, nobs.(iidsamples) .- 1)
config_samples = EmpirikosBNP.ConfigurationSample.(iidsamples)
muhat_scaled = mu_hats .* sqrt.(nobs.(iidsamples))


Zs_Ss = Empirikos.NormalChiSquareSample.(muhat_scaled, Ss)
# Run EPB:

epb =  Empirikos.EmpiricalPartiallyBayesTTest(;α=α_bh, solver=Mosek.Optimizer)
epb_fit = fit(epb, Zs_Ss)
epb_pvals = epb_fit.pvalue


# Partially Bayes method

neal2 = EmpirikosBNP.NealAlgorithm2(Ss)
neal2_samples = fit!(neal2; samples=10_000, burnin=5_000)

neal2_pvals = EmpirikosBNP._pval_fun(neal2_samples, muhat_scaled)


pt = PolyaTreeDistribution(base=Empirikos.fold(TDist(8)/std(TDist(8))), 
    J=8, α=1.0, 
    ρ = EmpirikosBNP.PTFun(;inner_multiplier=20.0, inner_power=2, 
        boundary_multiplier=0.1, boundary_power=0, symmetric=true),  #  ρ = (j,k) -> k == 2^(j-1) ? 0.5 : 20*j^2,
    symmetrized=true,
    median_centered=false)


neal8polya = EmpirikosBNP.NealAlgorithm8Polya(config_samples; base_polya=pt, neal_cp=deepcopy(neal2))
#neal_polya_samples = StatsBase.fit!(neal8polya; samples=10_000, burnin=2_000)
neal_polya_samples = StatsBase.fit!(neal8polya; samples=200, burnin=100)


neal_polya_pvals = EmpirikosBNP._pval_fun(neal_polya_samples, mu_hats; method=:monte_carlo)
oracle_pvals = EmpirikosBNP._pval_custom.(config_samples, mu_hats, σs, Ref(dbn); rtol=0.01)



results = (
    ttest = evaluate_pvals(Hs_bool, ttest_pvals),
    neal2 = evaluate_pvals(Hs_bool, neal2_pvals),
    neal_polya = evaluate_pvals(Hs_bool, neal_polya_pvals),
    oracle = evaluate_pvals(Hs_bool, oracle_pvals),
    sens_general = evaluate_rjs(Hs_bool, Bool.(SENS_rjs_general)),
    sens_gaussian = evaluate_rjs(Hs_bool, Bool.(SENS_rjs_gaussian)),
    epb = evaluate_pvals(Hs_bool, epb_pvals)
)


store_name = joinpath(dir, "simulation_results", "method_res_$(task_id).jld2")


jldsave(
    store_name;
    results = results,
    seed = seed,
    noise_dbn = dbn,
    subbotin_param = subbotin_param,  
    variance_dbn = variance_dbn,
    effect_size = effect_size,
    K = K,
    n = n,
    n1 = n1,
    task_id = task_id
)

if seed==1 
    store_name2 = joinpath(dir, "simulation_results", "method_res_$(task_id)_tree.jld2")
    jldsave(
        store_name2;
        neal_polya_samples = neal_polya_samples,
        neal2_samples = neal2_samples,
        iidsamples = iidsamples,
        μs = μs,
        σs = σs,
        Hs_bool
    )
end
