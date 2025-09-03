using Pkg
Pkg.activate(".")

using JLD2
using DataFrames
using Statistics
using Distributions
using CSV

function load_and_process_results(dir::String; task_ids=1:500)
    all_results = []
    
    for task_id in task_ids
        file_path = joinpath(dir, "simulation_results", "method_res_$(task_id).jld2")
        
        if isfile(file_path)
            data = load(file_path)
            
            push!(all_results, (
                task_id = task_id,
                seed = data["seed"],
                subbotin_param = data["subbotin_param"],
                variance_dbn = data["variance_dbn"],
                effect_size = data["effect_size"],
                K = data["K"],
                n = data["n"],
                n1 = data["n1"],
                results = data["results"]
            ))
        else
            @warn "File not found: $file_path"
        end
    end
    
    rows = []
    
    for res in all_results
        for (method_name, method_results) in pairs(res.results)
            push!(rows, (
                task_id = res.task_id,
                seed = res.seed,
                subbotin_param = res.subbotin_param,  
                variance_dbn = string(res.variance_dbn), 
                effect_size = res.effect_size,
                K = res.K,
                n = res.n,
                n1 = res.n1,
                method = string(method_name),
                
                # Metrics
                Uniformity_Pval = method_results.Uniformity_Pval,
                Power_Pval = method_results.Power_Pval,
                discoveries_Pval = method_results.discoveries_Pval,
                FDP_BH = method_results.FDP_BH,
                Power_BH = method_results.Power_BH,
                discoveries_BH = method_results.discoveries_BH
            ))
        end
    end
    
    df = DataFrame(rows)
    
    return df
end

function aggregate_results(df::DataFrame)

    metrics = [:Uniformity_Pval, :Power_Pval, :discoveries_Pval, 
               :FDP_BH, :Power_BH, :discoveries_BH]
    
    aggregated = combine(
        groupby(df, [:method, :subbotin_param, :variance_dbn]),
        [metric => mean => Symbol(string(metric) * "_mean") for metric in metrics]...,
        [metric => std => Symbol(string(metric) * "_std") for metric in metrics]...,
        nrow => :n_seeds
    )
    
    return aggregated
end

function process_simulation_results(dir::String; task_ids=1:500, include_std=false)
    println("Loading simulation results...")
    df_raw = load_and_process_results(dir; task_ids=task_ids)
    println("Loaded $(nrow(df_raw)) raw results")
    
    println("Aggregating results...")
    df_aggregated = aggregate_results(df_raw)
    println("Created $(nrow(df_aggregated)) aggregated rows")
    
    if !include_std
        cols_to_keep = [:method, :subbotin_param, :variance_dbn, :n_seeds]
        append!(cols_to_keep, [Symbol(string(m) * "_mean") for m in 
                              [:Uniformity_Pval, :Power_Pval, :discoveries_Pval, 
                               :FDP_BH, :Power_BH, :discoveries_BH]])
        df_aggregated = select(df_aggregated, cols_to_keep)
        
        rename_dict = Dict(Symbol(string(m) * "_mean") => m for m in 
                          [:Uniformity_Pval, :Power_Pval, :discoveries_Pval, 
                           :FDP_BH, :Power_BH, :discoveries_BH])
        rename!(df_aggregated, rename_dict)
    end
    
    sort!(df_aggregated, [:subbotin_param, :variance_dbn, :method])
    
    return df_aggregated
end

dir = @__DIR__
results_df = process_simulation_results(dir; task_ids=1:1000)
# 
CSV.write(joinpath(dir, "aggregated_results.csv"), results_df)
# 
# # Or save as JLD2
# jldsave(joinpath(dir, "aggregated_results.jld2"); results_df = results_df)