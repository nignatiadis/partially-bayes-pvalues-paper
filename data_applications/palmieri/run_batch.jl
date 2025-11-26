using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using EmpirikosBNP
using StatsBase
using JLD2
using Random

DIR = @__DIR__
CHECKPOINT_DIR = joinpath(DIR, "checkpoints")
BATCH_NUM = parse(Int, ARGS[1])
SAMPLES_PER_BATCH = 2_500

Random.seed!(BATCH_NUM)

println("Loading state for batch $BATCH_NUM...")
neal8polya = load(joinpath(CHECKPOINT_DIR, "neal8polya_state.jld2"), "neal8polya")

println("Running batch $BATCH_NUM ($(SAMPLES_PER_BATCH) samples)...")
samples = StatsBase.fit!(neal8polya; samples=SAMPLES_PER_BATCH, burnin=1)

jldsave(joinpath(CHECKPOINT_DIR, "samples_batch_$(BATCH_NUM).jld2"), samples = samples)
jldsave(joinpath(CHECKPOINT_DIR, "neal8polya_state.jld2"), neal8polya = neal8polya)

println("Batch $BATCH_NUM complete")