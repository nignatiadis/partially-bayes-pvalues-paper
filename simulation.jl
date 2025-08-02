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
K=5


variance_distributions = (
    Dirac = Dirac(1.0),
    Uniform = Uniform(0.5, 2.0)
)

noise_distributions = (
    subbotin_3 = PGeneralizedGaussian(0, 1, 3) / std(PGeneralizedGaussian(0, 1, 3)),
    normal = Normal(0, 1),  
    laplace = Laplace(0, 1) / std(Laplace(0, 1))
)

seeds = 1:50

key_combinations = collect(Iterators.product(variance_distributions, noise_distributions, seeds))

variance_dbn, dbn, seed = key_combinations[task_id]
effect_size = 2.5

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


ttests = OneSampleTTest.(response.(iidsamples))
ttest_pvals = pvalue.(ttests)


function evaluate_method(Hs, pvals; α_pval = 0.01, α_bh = 0.1)

    uniformity_pval = mean(pvals[.! Hs_bool] .<= α_pval) 
    power_pval = mean(pvals[Hs_bool] .<= α_pval)
    discoveries_pval = sum(pvals .<= α_pval)

    bh_pvals = MultipleTesting.adjust(pvals, BenjaminiHochberg())
    rj_idx = bh_pvals .<= α_bh
    discoveries = sum(rj_idx)
    true_discoveries = sum(rj_idx .& Hs)
    false_discoveries = discoveries - true_discoveries
    FDP = false_discoveries / max(discoveries, 1)
    Power = true_discoveries / max( sum(Hs), 1)

    (
        Uniformity_Pval = uniformity_pval,
        Power_Pval = power_pval,
        discoveries_Pval = discoveries_pval,
        FDP_BH = FDP, 
        Power_BH = Power, 
        discoveries_BH = discoveries
    )
end



Ss = Empirikos.ScaledChiSquareSample.(vars, nobs.(iidsamples) .- 1)
config_samples = EmpirikosBNP.ConfigurationSample.(iidsamples)

neal2 = EmpirikosBNP.NealAlgorithm2(Ss)
neal2_samples = fit!(neal2; samples=10_000, burnin=5_000)

muhat_scaled = mu_hats .* sqrt.(nobs.(iidsamples))
neal2_pvals = EmpirikosBNP._pval_fun(neal2_samples, muhat_scaled)




pt = PolyaTreeDistribution(base=base=Empirikos.fold(TDist(8)/std(TDist(8))), 
    J=8, α=1.0, 
    ρ = EmpirikosBNP.PTFun(;inner_multiplier=20.0, inner_power=2, 
        boundary_multiplier=0.1, boundary_power=0),  #  ρ = (j,k) -> k == 2^(j-1) ? 0.5 : 20*j^2,
    symmetrized=true,
    median_centered=false)


neal8polya = EmpirikosBNP.NealAlgorithm8Polya(config_samples; base_polya=pt, neal_cp=deepcopy(neal2))
neal_polya_samples = StatsBase.fit!(neal8polya; samples=5_000, burnin=1_000)


neal_polya_pvals = EmpirikosBNP._pval_fun(neal_polya_samples, mu_hats; method=:monte_carlo)



oracle_pvals = EmpirikosBNP._pval_custom.(config_samples, mu_hats, σs, Ref(dbn); rtol=0.01)

results = (
    ttest = evaluate_method(Hs_bool, ttest_pvals),
    neal2 = evaluate_method(Hs_bool, neal2_pvals),
    neal_polya = evaluate_method(Hs_bool, neal_polya_pvals),
    oracle = evaluate_method(Hs_bool, oracle_pvals)
)


dir = @__DIR__

store_name = joinpath(dir, "simulation_results", "method_res_$(task_id).jld2")
jldsave(
    store_name;
    results = results,
    seed = seed,
    noise_dbn = dbn,
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