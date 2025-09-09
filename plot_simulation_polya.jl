# Code to plot posterior Polya trees from some of the Monte Carlo replicates.


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
using Plots
using Test
using LaTeXStrings
pgfplotsx()

dir = @__DIR__

variance_distributions = (
    Dirac = Dirac(1.0),
    Uniform = Uniform(0.5, 2.0)
)

subbotin_parameters = (1.0, 1.5, 2.0, 2.5, 3.0)

seeds = 1:100

key_combinations = collect(Iterators.product(variance_distributions, subbotin_parameters, seeds))



#dirac_var_idx = [1, 3, 5, 7, 9]
#uniform_var_idx = [2, 4, 6, 8, 10]
dirac_var_idx = [1, 5, 9]
uniform_var_idx = [2, 6, 10]

for (idx_key, idx_set) in enumerate((dirac_var_idx, uniform_var_idx))
plots_list = []
for (idx, key_idx) in enumerate(idx_set)
    variance_dbn, subbotin_param, seed = key_combinations[key_idx]
    true_dbn = PGeneralizedGaussian(0, 1, subbotin_param) / std(PGeneralizedGaussian(0, 1, subbotin_param))
    
    file_path = joinpath(dir, "method_res_$(key_idx)_tree.jld2")
    data = load(file_path)
    neal_polya_samples = data["neal_polya_samples"]
    
    μs = -3:0.01:3
    densities = zeros(length(μs), 5000)
    for j in 1:5000
        pt_sample = neal_polya_samples.realized_pts[j]
        densities[:, j] = pdf(pt_sample, μs)
    end
    
    lower_q = [quantile(densities[i, :], 0.005) for i in 1:length(μs)]
    upper_q = [quantile(densities[i, :], 0.995) for i in 1:length(μs)]
    
    if idx == 1
        myplot = plot(μs, pdf(true_dbn, μs), 
             label="Truth", 
             lw=1.3, 
             color=:darkblue,
             alpha = 0.8,
             linestyle=:solid,
             title=L"\xi = %$(subbotin_param)",
             grid = true,
             gridalpha = 0.3,
             gridstyle = :dot,
             framestyle = :box,
             legend=:topright,
             xlabel=L"t",
             ylabel=L"w(t)",
             ylim = (0,0.8)
             )  
    else
        myplot = plot(μs, pdf(true_dbn, μs), 
             label=false,
             lw=1.3, 
             color=:darkblue,
             alpha = 0.8,
             title=L"\xi = %$(subbotin_param)",
             grid = true,
             gridalpha = 0.3,
             gridstyle = :dot,
             framestyle = :box,
             legend=false,
             xlabel=L"t",
             ylabel="",
             ylim = (0,0.8)
             )
    end
    
    if idx == 1
        plot!(myplot, μs, lower_q, 
              fillrange=upper_q,
              fillalpha=0.55,
              linealpha=0,
              color=:gray,
              label="99% CI")
    else
        plot!(myplot, μs, lower_q, 
              fillrange=upper_q,
              fillalpha=0.55,
              linealpha=0,
              lw=0,
              color=:gray,
              label=false)
    end
    
    pt_final = neal_polya_samples.realized_pts[5000]
    if idx == 1
        plot!(myplot, μs, pdf(pt_final, μs), 
              lw=1, 
              color=:green,
              alpha=0.6,
              label="Post.")
    else
        plot!(myplot, μs, pdf(pt_final, μs), 
              lw=1, 
              color=:green,
              alpha=0.6,
              label=false)
    end
    
    push!(plots_list, myplot)
end

combined_plot = plot(plots_list..., 
                     layout=(1, 3), 
                     size=(1000, 300),
                     left_margin=0Plots.mm,
                     right_margin=-1Plots.mm,
                     bottom_margin=3Plots.mm,
                     top_margin=2Plots.mm,
                     spacing=0Plots.mm)

combined_plot

savefig(combined_plot, "posterior_polya_trees_$(idx_key).pdf")
end