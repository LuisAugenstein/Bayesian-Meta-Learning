#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=16000
#SBATCH --time=8:00:00
#SBATCH --parsable

# Testing PLATIPUS with params from the paper
# Not many hyperparams provided, assumed to be similar to MAML

echo 'PLATIPUS NLML Test started'

EPOCHS=50000

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done

for benchmark in Sinusoid1D
do
    for num_samples in 5
    do
        for num_inner_updates in 1 10
        do
            for seed in 123
            do
                python train.py --algorithm platipus \
                                --runner nlml_runner \
                                --wandb True \
                                --num_epochs $EPOCHS \
                                --benchmark $benchmark \
                                --num_models 10 \
                                --k_shot $num_samples \
                                --seed $seed \
                                --inner_lr 0.01 \
                                --meta_lr 0.001 \
                                --minibatch 6 \
                                --num_episodes_per_epoch 6 \
                                --num_test_tasks 20 \
                                --noise_stddev 0.02 \
                                --num_hidden 2 \
                                --hidden_size 40 \
                                --num_inner_updates $num_inner_updates \
                                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta
            done
        done
    done
done