using Empirikos
using EmpirikosBNP
using CSV
using LaTeXStrings
using Statistics
using Plots
using Setfield
using MosekTools   
using MultipleTesting 
using DataFrames
using HypothesisTests
using StatsBase
using Distributions
using JLD2
dir = @__DIR__
using LaTeXStrings
using Test
using Random


pgfplotsx()
theme(
    :default;
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    gridalpha = 0.3,
    gridstyle = :dot,
    framestyle = :box,
    legendfonthalign = :left,
    thickness_scaling = 1.3,
    size = (420, 330),
)


microarray = CSV.File(joinpath(dir, "palmieri_pairwise.csv")) |>
    DataFrame


identifiers = names(microarray)[4:end]


diffs = Matrix(microarray[:, identifiers])


ttests = OneSampleTTest.(eachrow(diffs))
ttest_pvals = pvalue.(ttests)
sum(ttest_pvals .<= 0.001)

# Prepare for use with EmpirikosBNP
iidsamples = EmpirikosBNP.IIDSample.(eachrow(diffs))
vars = var.(EmpirikosBNP.iid_samples.(iidsamples)) 
mu_hats = mean.(EmpirikosBNP.iid_samples.(iidsamples))
Ss = Empirikos.ScaledChiSquareSample.(vars, nobs.(iidsamples) .- 1)
config_samples = EmpirikosBNP.ConfigurationSample.(iidsamples)

# Sanity check 
ttest_pvals2 = 2*ccdf.(TDist.(11), abs.(mu_hats) ./ (sqrt.(vars) ./ sqrt(12)) )
@test ttest_pvals == ttest_pvals2

Random.seed!(1)


# Run Partially Bayes methods assuming normality
neal2 = EmpirikosBNP.NealAlgorithm2(Ss)
neal2_samples = fit!(neal2; samples=100_000, burnin=5_000)


muhat_scaled = mu_hats .* sqrt.(nobs.(iidsamples))
neal2_pvals = EmpirikosBNP._pval_fun(neal2_samples, muhat_scaled)

sum(neal2_pvals .<= 0.001)



pt = PolyaTreeDistribution(base=base=Empirikos.fold(TDist(8)/std(TDist(8))), 
    J=8, α=1.0, 
    ρ = EmpirikosBNP.PTFun(;inner_multiplier=20.0, inner_power=2, 
        boundary_multiplier=0.1, boundary_power=0),  #  ρ = (j,k) -> k == 2^(j-1) ? 0.5 : 20*j^2,
    symmetrized=true,
    median_centered=false)

Random.seed!(1)

samples_per_run = 5_000
base_filename = "August1_neal8polya"
neal8polya = EmpirikosBNP.NealAlgorithm8Polya(config_samples; base_polya=pt, neal_cp=deepcopy(neal2))


run_sims = false 

if run_sims

_ = StatsBase.fit!(neal8polya; samples=1, burnin=2_000) # first batch only for burnin
jldsave("August1_neal8polya_burnin.jld2", neal8polya = neal8polya)


for i in 1:20
    println("Running iteration $(i)...")
    
    # Run the sampling
    neal8polya_samples = StatsBase.fit!(neal8polya; 
                                       samples=samples_per_run, 
                                       burnin=1)
    
    # Save checkpoint
    filename = "$(base_filename)_$(i).jld2"
    jldsave(filename, 
            neal8polya = neal8polya,
            neal8polya_samples = neal8polya_samples)
    println("Saved checkpoint: $filename")
end
end


all_samples = []

for i in 1:20
    filename = "$dir/$(base_filename)_$(i).jld2"
    data = load(filename)
    push!(all_samples, data["neal8polya_samples"])
end

merged_samples = EmpirikosBNP._merge_samples(all_samples...)
all_samples = []; # free memory
GC.gc()
neal_polya_pvals = EmpirikosBNP._pval_fun(merged_samples, mu_hats; method=:monte_carlo)

@test size(merged_samples.Zs_mat,2) == 20 * samples_per_run

### Now we have all three sets of p-values. 

### Rejections for each

sum(ttest_pvals .<= 0.001)
sum(neal2_pvals .<= 0.001)
sum(neal_polya_pvals .<= 0.001)

### Let us plot the corresponding qq-plots



function plot_pvalues_qq(p1, p2, p3, names=("Method 1", "Method 2", "Method 3"))
    n = length(p1)
    # Thin out the points - take every kth point
    k = 1  # adjust this to control density
    idx = 1:k:n
    theoretical_q = range(0, 1, length=n+1)[2:end][idx]
    
    breaks = 10.0 .^ (0:-1:-5)
    
    colors = ("#E69F00", "#56B4E9", "#009E73")
    
    p = plot(theoretical_q, sort(p1)[idx], 
            label=names[1], 
            xlabel=L"Uniform quantile $(P_i)$", 
            ylabel=L"Empirical quantile $(P_i)$",
            xscale=:log10, 
            yscale=:log10,
            xticks=breaks,
            yticks=breaks,
            color=colors[1],
            linewidth=1.7,
            legend=:topleft,
            size=(500, 450),
            )
            
    plot!(theoretical_q, sort(p2)[idx], 
         label=names[2], 
         color=colors[2],
         linewidth=1.7)
         
    plot!(theoretical_q, sort(p3)[idx], 
         label=names[3], 
         color=colors[3],
         linewidth=1.7)
         
    plot!([minimum(theoretical_q), 1], [minimum(theoretical_q), 1], 
          color=:gray, 
          linestyle=:dash, 
          label="Uniform",
          linewidth=1.3)
    
    count1 = sum(p1 .<= 0.001)
    count2 = sum(p2 .<= 0.001)
    count3 = sum(p3 .<= 0.001)

    annotate!(0.0095, 0.0005/4, text(L"\# P_i \leq 0.001", :left, 10))
    annotate!(0.01, 0.0002/4, text("$(names[1]): $count1", :left, 10))
    annotate!(0.01, 0.0001/4, text("$(names[2]): $count2", :left, 10))
    annotate!(0.01, 0.000045/4, text("$(names[3]): $count3", :left, 10))

    hline!([0.001], linestyle=:dot, label=L"\alpha=0.001", color=:black)

    return p
end

plot_pvalues_qq(ttest_pvals, neal2_pvals, neal_polya_pvals, 
    ("t-test", "Normal PB", "Pólya PB"))
savefig("qqplot_palmieri.pdf")


## Plot the two-dimensional histogram and show 2D summarization



vars_flat =  log.(vars)



α_quantile = 0.999
α_quantile_twosided = 0.9995
qs = [quantile(abs.(row), α_quantile) for row in eachrow(merged_samples.Zs_mat)]

function _quantile_fun(samples::EmpirikosBNP.NealAlgorithmSamples, alpha::Float64=0.999)
    n_rows = size(samples.assignments, 1)
    quantile_mat = Vector{Float64}(undef, n_rows)
    
    for i in 1:n_rows
        # Get component parameters across all samples for this row i
        sigmas = [sqrt(getproperty(samples.components[j][samples.assignments[i,j]], :param)) 
                 for j in 1:size(samples.assignments, 2)]
        
        # Create mixture model with Normal(0, sigma) components
        components = [Normal(0, sigma) for sigma in sigmas]
        weights = fill(1/length(components), length(components))
        mixture = MixtureModel(components, weights)
        
        # Compute quantile
        quantile_mat[i] = quantile(mixture, alpha)
    end
    
    quantile_mat
end

qs_neal = _quantile_fun(neal2_samples, α_quantile_twosided)

sorted_indices = sortperm(vars_flat)
sorted_qs = qs[sorted_indices]
sorted_qs_neal = qs_neal[sorted_indices]
sorted_vars = vars_flat[sorted_indices]
isoreg = MultipleTesting.isotonic_regression(sorted_qs)
isoreg_neal = MultipleTesting.isotonic_regression(sorted_qs_neal)

#idxs = 2:10:length(sorted_vars)
#grid = sorted_vars[idxs]
grid = sorted_vars


oracle_t = sqrt.(exp.(grid)) .* quantile(TDist(11), α_quantile_twosided) ./ sqrt(12)

oracle_t2 = sqrt.(exp.(vars_flat)) .* quantile(TDist(11),  α_quantile_twosided) ./ sqrt(12)
@test sum(ttest_pvals .<= 0.001) == sum(abs.(mu_hats) .>= oracle_t2)

sum(ttest_pvals .<= 0.001) 
sum(abs.(mu_hats[sorted_indices]) .>= oracle_t)

sum( abs.(mu_hats[sorted_indices]) .>= isoreg_neal ./ sqrt(12))
sum(neal2_pvals .<= 0.001)

sum( abs.(mu_hats[sorted_indices]) .>= isoreg)
sum(neal_polya_pvals .<= 0.001)


log_grid = [0.01; 0.1; 1.0; 10.0]


cutoff_matrix =
    [oracle_t isoreg_neal ./ sqrt(12) isoreg]
cutoff_colors = ["#E69F00" "#56B4E9" "#009E73"] 
cutoff_linestyles = [:dash :solid :solid]


twod_histogram_plot = histogram2d(
    vars_flat,
    mu_hats,
    bins = 50,
    c = cgrad(:algae, rev = false, scale = :exp),
    xlabel = L"S_i^2",
    ylabel = L"\bar{Z}_i",
    xticks = (log.(log_grid), string.(log_grid)),
    ylim = (-3.5,3.5),
    size = (500,450)
)

plot!(
    twod_histogram_plot,
    grid,
    cutoff_matrix,
    color = cutoff_colors,
    linestyle = cutoff_linestyles,
    label = ["t-test" "Normal PB" "Pólya PB"],
)

plot!(
    twod_histogram_plot,
    grid,
    -cutoff_matrix,
    color = cutoff_colors,
    linestyle = cutoff_linestyles,
    label = "",
)

savefig("twod_histogram_palmieri.pdf")





# Further visualization

plot(length.(unique.(eachcol(neal2_samples.assignments))))



all_σs = [pt.σ for pt in merged_samples.realized_pts]
all_σs = [std(pt) for pt in merged_samples.realized_pts]