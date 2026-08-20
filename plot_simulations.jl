using Pkg
Pkg.activate()
using CSV
using DataFrames
using Plots
using LaTeXStrings
pgfplotsx()  
push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepackage{amssymb}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepackage{amsmath}")

df = CSV.read("aggregated_results.csv", DataFrame)

# check if EPB ~ Normal PB

df_epb = filter(row -> row.method == "epb", df)
df_neal2 = filter(row -> row.method == "neal2", df)

if nrow(df_epb) > 0 && nrow(df_neal2) > 0
    merged = innerjoin(df_epb, df_neal2, 
        on = [:subbotin_param, :variance_dbn],  # adjust join keys as needed
        makeunique = true)
    for metric in ["Uniformity_Pval", "Power_Pval", "FDP_BH", "Power_BH"]
        col1 = Symbol(metric)
        col2 = Symbol(metric * "_1")
        if hasproperty(merged, col1) && hasproperty(merged, col2)
            vals1 = collect(skipmissing(merged[!, col1]))
            vals2 = collect(skipmissing(merged[!, col2]))
            if length(vals1) > 0 && length(vals2) > 0
                max_diff = maximum(abs.(vals1 .- vals2) ./ vals1)
                println("$metric max |epb - neal2| = $max_diff")
            end
        end
    end
end



method_colors = Dict(
    "neal2" => RGB(0.0, 0.447, 0.741),       # Blue
    "neal_polya" => RGB(0.850, 0.325, 0.098), # Orange  
    "oracle" => RGB(0.929, 0.694, 0.125),     # Yellow/Gold
    "ttest" => RGB(0.494, 0.184, 0.556),      # Purple
    "sens_gaussian" => RGB(0.466, 0.674, 0.188)        # Green
)

method_styles = Dict(
    "neal2" => :solid,
    "neal_polya" => :dash,
    "oracle" => :dashdot,
    "ttest" => :dot,
    "sens_gaussian" => :dashdotdot
)

method_markers = Dict(
    "neal2" => :circle,
    "neal_polya" => :square,
    "oracle" => :diamond,
    "ttest" => :utriangle,
    "sens_gaussian" => :star5
)

method_names = Dict(
    "neal2" => "EPB & Normal PB",
    "neal_polya" => "Pólya PB",
    "oracle" => "oracle",
    "ttest" => "t-test",
    "sens_gaussian" => "SENS"
)

# Define which methods to use for each metric type
pval_methods = ["neal2", "neal_polya", "oracle", "ttest"]  # Methods with valid p-values
fdr_methods = ["neal2", "neal_polya", "oracle", "ttest", "sens_gaussian"]  # All methods for FDR control

metrics = ["Uniformity_Pval", "Power_Pval", "FDP_BH", "Power_BH"]
y_labels = [
    L"\mathbb{P}[P_i \leq 0.01 \mid \theta_i=0]",
    L"\text{Power}[P_i \leq 0.01]",
    L"\text{FDR}[\text{BH}]",
    L"\text{Power}[\text{BH}]"
]

variance_dbns = unique(df.variance_dbn)

function create_row_plot(var_dbn, row_idx)
    plots_array = []
    
    df_var = filter(row -> row.variance_dbn == var_dbn, df)
    
    for (col_idx, (metric, y_label)) in enumerate(zip(metrics, y_labels))
        
        # Determine which methods to plot for this metric
        is_pval_metric = metric in ["Uniformity_Pval", "Power_Pval"]
        methods_to_plot = is_pval_metric ? pval_methods : fdr_methods
        
        # Filter data for relevant methods and compute y_max
        df_methods = filter(row -> row.method in methods_to_plot, df_var)
        metric_data = df_methods[!, metric]
        metric_data = collect(skipmissing(metric_data))
        y_max = length(metric_data) > 0 ? maximum(metric_data) * 1.02 : 1.0
        
        # Show legend in column 1 (pval methods) and column 3 (fdr methods)
        show_legend = (row_idx == 1) && (col_idx == 1 || col_idx == 3)
        
        p = plot(
            xlabel = L"\xi",
            ylabel = y_label,
            title = "",  # No title
            xlims = (0.8, 3.2),
            ylims = (0, y_max),
            xticks = 1:0.5:3,
            legend = show_legend ? :bottomright : false,
            background_color_legend = nothing,
            foreground_color_legend = nothing,
            grid = true,
            gridalpha = 0.3,
            gridstyle = :dot,
            titlefontsize = 11,
            guidefontsize = 10,
            tickfontsize = 8,
            legendfontsize = 7,
            framestyle = :box
        )
        
        for method in methods_to_plot
            method_data = filter(row -> row.method == method, df_var)
            sort!(method_data, :subbotin_param)
            
            if nrow(method_data) == 0
                continue
            end
            
            plot!(p, 
                  method_data.subbotin_param, 
                  method_data[!, metric],
                  label = method_names[method],
                  color = method_colors[method],
                  linestyle = method_styles[method],
                  marker = method_markers[method],
                  markersize = 4,
                  linewidth = 2,
                  markerstrokewidth = 0)
        end
        
        if metric == "Uniformity_Pval"
            hline!(p, [0.01], 
                   linestyle = :dash, 
                   color = :gray, 
                   label = "", 
                   linewidth = 1.5,
                   alpha = 0.7)
        elseif metric == "FDP_BH"
            hline!(p, [0.1], 
                   linestyle = :dash, 
                   color = :gray, 
                   label = "", 
                   linewidth = 1.5,
                   alpha = 0.7)
        end
        
        push!(plots_array, p)
    end
    
    return plots_array
end

# Dirac variances
plots_row1 = create_row_plot(variance_dbns[1], 1)
plot_row1 = plot(plots_row1..., 
                 layout = (1, 4), 
                 size = (1100, 270),
                 left_margin = 1Plots.mm,
                 right_margin = 0.5Plots.mm,
                 bottom_margin = 8Plots.mm,
                 top_margin = 2Plots.mm,
                 wspace = 0.00)  

# Uniform variances
plots_row2 = create_row_plot(variance_dbns[2], 2)
plot_row2 = plot(plots_row2..., 
                 layout = (1, 4), 
                 size = (1100, 270),
                 left_margin = 1Plots.mm,
                 right_margin = 0.5Plots.mm,
                 bottom_margin = 8Plots.mm,
                 top_margin = 2Plots.mm,
                 wspace = 0.00)  

# Display the plots
display(plot_row1)
display(plot_row2)

# Save the plots
savefig(plot_row1, "simulation_results_row1_dirac.pdf")
savefig(plot_row2, "simulation_results_row2_uniform.pdf")