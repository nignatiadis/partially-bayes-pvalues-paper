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

method_colors = Dict(
    "neal2" => RGB(0.0, 0.447, 0.741),      # Blue
    "neal_polya" => RGB(0.850, 0.325, 0.098), # Orange  
    "oracle" => RGB(0.929, 0.694, 0.125),    # Yellow/Gold
    "ttest" => RGB(0.494, 0.184, 0.556)      # Purple
)

method_styles = Dict(
    "neal2" => :solid,
    "neal_polya" => :dash,
    "oracle" => :dashdot,
    "ttest" => :dot
)

method_markers = Dict(
    "neal2" => :circle,
    "neal_polya" => :square,
    "oracle" => :diamond,
    "ttest" => :utriangle
)

method_names = Dict(
    "neal2" => "Normal PB",
    "neal_polya" => "Pólya PB",
    "oracle" => "oracle",
    "ttest" => "t-test"
)

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
        
        metric_data = df_var[!, metric]
        y_max = maximum(metric_data) * 1.1  
        
        p = plot(
            xlabel = L"\xi",
            ylabel = y_label,
            title = "",  # No title
            xlims = (0.8, 3.2),
            ylims = (0, y_max),
            xticks = 1:0.5:3,
            legend = (row_idx == 1 && col_idx == 1) ? :bottomright : false,
            grid = true,
            gridalpha = 0.3,
            gridstyle = :dot,
            titlefontsize = 11,
            guidefontsize = 10,
            tickfontsize = 8,
            legendfontsize = 8,
            framestyle = :box
        )
        
        for method in unique(df.method)
            method_data = filter(row -> row.method == method, df_var)
            sort!(method_data, :subbotin_param)
            
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
#savefig(plot_row1, "simulation_results_row1_dirac.png", dpi=300)
savefig(plot_row2, "simulation_results_row2_uniform.pdf")
#savefig(plot_row2, "simulation_results_row2_uniform.png", dpi=300)

