using Pkg
Pkg.activate(".")

using Empirikos
using EmpirikosBNP
using LaTeXStrings
using Statistics
using Plots
using MultipleTesting 
using StatsBase
using Distributions
using JLD2
using ProgressMeter

const DIR = @__DIR__
const CHECKPOINT_DIR = joinpath(DIR, "checkpoints")
const OUTPUT_DIR = joinpath(DIR, "output")
const N_BATCHES = 40  # 40 batches × 2500 MC = 100,000 MC total
mkpath(OUTPUT_DIR)

pgfplotsx()
theme(:default;
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    gridalpha = 0.3, gridstyle = :dot, framestyle = :box,
    legendfonthalign = :left, thickness_scaling = 1.3, size = (420, 330),
)

# Load data
neal2_data = load(joinpath(CHECKPOINT_DIR, "neal2_results.jld2"))
mu_hats = neal2_data["mu_hats"]
vars = neal2_data["vars"]
ttest_pvals = neal2_data["ttest_pvals"]
neal2_samples = neal2_data["neal2_samples"];

neal2_pvals = EmpirikosBNP._pval_fun(neal2_samples, mu_hats .* sqrt(12))

# Merge batches
println("Merging $N_BATCHES batches")
all_samples = [load(joinpath(CHECKPOINT_DIR, "samples_batch_$i.jld2"), "samples") for i in 1:N_BATCHES]
merged_samples = EmpirikosBNP._merge_samples(all_samples...);
all_samples = nothing; GC.gc()

neal_polya_pvals = EmpirikosBNP._pval_fun(merged_samples, mu_hats; method=:monte_carlo)

jldsave(joinpath(OUTPUT_DIR, "final_pvalues.jld2"),
    ttest_pvals = ttest_pvals,
    neal2_pvals = neal2_pvals,
    neal_polya_pvals = neal_polya_pvals
)

println("Rejections at α=0.001:")
println("  t-test:    $(sum(ttest_pvals .<= 0.001))")
println("  Normal PB: $(sum(neal2_pvals .<= 0.001))")
println("  Pólya PB:  $(sum(neal_polya_pvals .<= 0.001))")

# QQ plot
function plot_pvalues_qq(p1, p2, p3, names)
    n = length(p1)
    theoretical_q = range(0, 1, length=n+1)[2:end]
    breaks = 10.0 .^ (0:-1:-5)
    colors = ("#E69F00", "#56B4E9", "#009E73")
    
    p = plot(theoretical_q, sort(p1), label=names[1], 
        xlabel=L"Uniform quantile $(P_i)$", ylabel=L"Empirical quantile $(P_i)$",
        xscale=:log10, yscale=:log10, xticks=breaks, yticks=breaks,
        color=colors[1], linewidth=1.7, legend=:topleft, size=(500, 450))
    plot!(theoretical_q, sort(p2), label=names[2], color=colors[2], linewidth=1.7)
    plot!(theoretical_q, sort(p3), label=names[3], color=colors[3], linewidth=1.7)
    plot!([minimum(theoretical_q), 1], [minimum(theoretical_q), 1], 
        color=:gray, linestyle=:dash, label="Uniform", linewidth=1.3)
    
    annotate!(0.0095, 0.0005/4, text(L"\# P_i \leq 0.001", :left, 10))
    annotate!(0.01, 0.0002/4, text("$(names[1]): $(sum(p1 .<= 0.001))", :left, 10))
    annotate!(0.01, 0.0001/4, text("$(names[2]): $(sum(p2 .<= 0.001))", :left, 10))
    annotate!(0.01, 0.000045/4, text("$(names[3]): $(sum(p3 .<= 0.001))", :left, 10))
    hline!([0.001], linestyle=:dot, label=L"\alpha=0.001", color=:black)
    p
end

plot_pvalues_qq(ttest_pvals, neal2_pvals, neal_polya_pvals, ("t-test", "Normal PB", "Pólya PB"))
savefig(joinpath(OUTPUT_DIR, "qqplot_palmieri.pdf"))

# 2D histogram
vars_flat = log.(vars)
α_quantile_twosided = 0.9995
qs = [quantile(abs.(row), 0.999) for row in eachrow(merged_samples.Zs_mat)]

function _quantile_fun(samples::EmpirikosBNP.NealAlgorithmSamples, alpha::Float64=0.999)
    n_rows = size(samples.assignments, 1)
    quantile_mat = Vector{Float64}(undef, n_rows)
    
    @showprogress for i in 1:n_rows
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
isoreg = MultipleTesting.isotonic_regression(qs[sorted_indices])
isoreg_neal = MultipleTesting.isotonic_regression(qs_neal[sorted_indices])
grid = vars_flat[sorted_indices]
oracle_t = sqrt.(exp.(grid)) .* quantile(TDist(11), α_quantile_twosided) ./ sqrt(12)

twod_histogram_plot = histogram2d(vars_flat, mu_hats, bins=50,
    c=cgrad(:algae, rev=false, scale=:exp), xlabel=L"S_i^2", ylabel=L"\bar{Z}_i",
    xticks=(log.([0.01, 0.1, 1.0, 10.0]), ["0.01", "0.1", "1.0", "10.0"]),
    ylim=(-3.5, 3.5), size=(500, 450))

for (cutoff, color, style, label) in [
    (oracle_t, "#E69F00", :dash, "t-test"),
    (isoreg_neal./sqrt(12), "#56B4E9", :solid, "Normal PB"),
    (isoreg, "#009E73", :solid, "Pólya PB")
]
    plot!(twod_histogram_plot, grid, cutoff, color=color, linestyle=style, label=label, linewidth=1.7)
    plot!(twod_histogram_plot, grid, -cutoff, color=color, linestyle=style, label="", linewidth=1.7)
end
twod_histogram_plot

savefig(joinpath(OUTPUT_DIR, "twod_histogram_palmieri.pdf"))
