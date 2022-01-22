#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=16000
#SBATCH --time=8:00:00
#SBATCH --parsable

# Testing PLATIPUS with params from the paper
# Not many hyperparams provided, assumed to be similar to MAML

echo 'PLATIPUS started'

EPOCHS=70000
EPOCHS_TO_STORE=2000

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done

for benchmark in Sinusoid1D
do
    for num_samples in 1 5 10
    do
        for num_inner_updates in 1 10
        do
            for seed in 1234 4321 9999 889 441 588 7741
            do
                python train.py --algorithm platipus \
                                --wandb True \
                                --num_epochs $EPOCHS \
                                --benchmark $benchmark \
                                --num_models 10 \
                                --k_shot $num_samples \
                                --seed $seed \
                                --seed_offset $seed \
                                --seed_offset_test $seed \
                                --inner_lr 0.01 \
                                --meta_lr 0.001 \
                                --minibatch 25 \
                                --noise_stddev 0.3 \
                                --num_hidden 2 \
                                --hidden_size 40 \
                                --num_inner_updates $num_inner_updates \
                                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta
            done
        done
    done
done