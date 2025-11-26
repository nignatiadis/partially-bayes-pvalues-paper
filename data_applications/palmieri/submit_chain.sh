#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"
mkdir -p logs

N_BATCHES=40

JOB0=$(sbatch --parsable 0_burnin.sbatch)
echo "Burnin: $JOB0"

PREV=$JOB0
for i in $(seq 1 $N_BATCHES); do
    JOB=$(sbatch --parsable --dependency=afterok:$PREV 1_batch.sbatch $i)
    echo "Batch $i: $JOB"
    PREV=$JOB
done

JOB_FINAL=$(sbatch --parsable --dependency=afterok:$PREV 2_analyze.sbatch)
echo "Analyze: $JOB_FINAL"

echo ""
echo "Monitor: squeue -u $USER"