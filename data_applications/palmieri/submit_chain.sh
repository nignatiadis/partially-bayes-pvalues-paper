#!/bin/bash


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


echo ""
echo "Monitor: squeue -u $USER"
