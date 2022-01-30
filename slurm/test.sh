#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=16000
#SBATCH --time=24:00:00
#SBATCH --parsable

# Testing PLATIPUS with params from the paper
# Not many hyperparams provided, assumed to be similar to MAML

echo 'PLATIPUS started'

EPOCHS=1

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done

python -W ignore train.py
