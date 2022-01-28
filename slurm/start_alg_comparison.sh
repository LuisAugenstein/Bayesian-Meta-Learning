#!/bin/bash

echo 'COMPARISON started'

EPOCHS=20000

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done

for benchmark in Sinusoid1D
do
    for alg in baseline maml platipus bmaml
    do
        sbatch slurm/alg_comparison.sh ALGO=$alg BENCH=$benchmark
    done
done