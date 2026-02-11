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
using QuadGK
using LogarithmicNumbers

dir = @__DIR__



n = 10_000
n1 = 1_000 #alternative proportion
K = 12
α_bh = 0.1
α_pval = 0.01



seeds = 1:100



seed = seeds[task_id]
effect_size = 1.5

σ = 1.0
dbn_uncentered = Exponential(1.0) 
dbn = dbn_uncentered - median(dbn_uncentered)



Random.seed!(1)
Zs_mat_init =  rand(dbn, n, K)
Hs = sample(1:n,n1, replace=false)
for i in Base.OneTo(seed)
    Zs_mat_init .= rand(dbn, n, K)
    Hs .= sample(1:n,n1, replace=false)
end
Hs_bool = falses(n)
Hs_bool[Hs] .= true
μs = zeros(n)
μs[Hs_bool] .= effect_size ./ sqrt(K)
Zs_mat = Zs_mat_init .+ μs

iidsamples = EmpirikosBNP.IIDSample.(collect.(eachrow(Zs_mat)))
mu_hats = mean.(getfield.(iidsamples, :Z))
median_hats = median.(getfield.(iidsamples, :Z))
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

function _pval_custom2(config_sample, z, σ, realized_pt; rtol=0.01)
    # Scale the Polya tree by σ
    scaled_pt = realized_pt * σ  #/ std(realized_pt)=1
    
    # Define density function
    t_density(t) = exp(ULogarithmic, logpdf(scaled_pt, config_sample, t))
    
    lb =  -median(dbn_uncentered) - minimum(config_sample.configuration) 

    # Normalize
    norm_const = QuadGK.quadgk(t_density, lb, Inf; rtol=rtol)[1]
    
    # Normalized density
    normalized_density(t) = t_density(t) / norm_const   
    
    # Two-tailed p-value
    p_upper = QuadGK.quadgk(normalized_density, abs(z), Inf; rtol=rtol)[1]
    p_lower = QuadGK.quadgk(normalized_density, lb, -abs(z); rtol=rtol)[1]
    min(Float64(p_upper + p_lower), 1.0)
end


ttests = OneSampleTTest.(response.(iidsamples))
ttest_pvals = pvalue.(ttests)

wilcoxon_tests = ExactSignedRankTest.(response.(iidsamples))
wilcoxon_pvals = min.(pvalue.(wilcoxon_tests), 1)

Ss = Empirikos.ScaledChiSquareSample.(vars, nobs.(iidsamples) .- 1)
config_samples = EmpirikosBNP.ConfigurationSample.(iidsamples)


neal2 = EmpirikosBNP.NealAlgorithm2(Ss)
neal2_samples = fit!(neal2; samples=10_000, burnin=5_000)

muhat_scaled = mu_hats .* sqrt.(nobs.(iidsamples))
neal2_pvals = EmpirikosBNP._pval_fun(neal2_samples, muhat_scaled)


pt = PolyaTreeDistribution(base=TDist(8)/std(TDist(8)), 
    J=8, α=1.0, 
    ρ = EmpirikosBNP.PTFun(;inner_multiplier=20.0, inner_power=2, 
        boundary_multiplier=0.1, boundary_power=0, symmetric = false),  #  ρ = (j,k) -> k == 2^(j-1) ? 0.5 : 20*j^2,
    symmetrized=false,
    median_centered=true)


neal8polya = EmpirikosBNP.NealAlgorithm8Polya(config_samples; base_polya=pt, neal_cp=deepcopy(neal2))
neal_polya_samples = StatsBase.fit!(neal8polya; samples=10_000, burnin=2_000)

neal_polya_pvals= EmpirikosBNP._pval_fun_median(neal_polya_samples, mu_hats, median_hats)

config_samples_median = EmpirikosBNP.ConfigurationSample.(collect.(eachrow(Zs_mat .- median_hats)), 1, median_hats)  
oracle_pvals = _pval_custom2.(config_samples_median, median_hats, 1.0, Ref(dbn); rtol=1e-3)









results = (
    ttest = evaluate_pvals(Hs_bool, ttest_pvals),
    wilcoxon = evaluate_pvals(Hs_bool, wilcoxon_pvals),
    neal2 = evaluate_pvals(Hs_bool, neal2_pvals),
    neal_polya = evaluate_pvals(Hs_bool, neal_polya_pvals),
    oracle = evaluate_pvals(Hs_bool, oracle_pvals)
)

store_name = joinpath(dir, "simulation_results", "median", "method_res_$(task_id).jld2")


jldsave(
    store_name;
    results = results,
    seed = seed,
    dbn_uncentered = dbn_uncentered,
    dbn = dbn,
    effect_size = effect_size,
    K = K,
    n = n,
    n1 = n1,
    task_id = task_id
)

if seed==1 
    store_name2 = joinpath(dir, "simulation_results", "median", "method_res_$(task_id)_tree.jld2")
    jldsave(
        store_name2;
        neal_polya_samples = neal_polya_samples,
        neal2_samples = neal2_samples,
        iidsamples = iidsamples,
        μs = μs,
        Hs_bool
    )
end



