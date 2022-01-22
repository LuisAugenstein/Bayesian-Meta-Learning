#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=32000
#SBATCH --time=16:00:00
#SBATCH --parsable

# Testing BMAML with params from the paper
# Sinsoid is simplified compared to the paper to retain
# comparability across implementations

echo 'BMAML started'

EPOCHS=100000
EPOCHS_TO_STORE=10000

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done

for benchmark in Sinusoid1D
do
    for num_samples in 5 10
    do
        for particles in 5 10
        do
            for seed in 1234 4321 9999 889 441 588 7741
            do
                python train.py --algorithm bmaml \
                                --wandb True \
                                --num_epochs 100000 \
                                --num_train_tasks 100 \
                                --benchmark $benchmark \
                                --num_models $particles \
                                --k_shot $num_samples \
                                --num_episodes_per_epoch $EPOCHS_TO_STORE \
                                --seed $seed \
                                --seed_offset $seed \
                                --seed_offset_test $seed \
                                --inner_lr 0.01 \
                                --meta_lr 0.01 \
                                --minibatch 10 \
                                --noise_stddev 0.3 \
                                --num_hidden 3 \
                                --hidden_size 40 \
                                --num_inner_updates 1 \
                                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta

                python train.py --algorithm bmaml \
                                --wandb True \
                                --num_epochs 100000 \
                                --num_train_tasks 1000 \
                                --benchmark $benchmark \
                                --num_models $particles \
                                --k_shot $num_samples \
                                --num_episodes_per_epoch $EPOCHS_TO_STORE \
                                --seed $seed \
                                --seed_offset $seed \
                                --seed_offset_test $seed \
                                --inner_lr 0.01 \
                                --meta_lr 0.01 \
                                --minibatch 10 \
                                --noise_stddev 0.3 \
                                --num_hidden 3 \
                                --hidden_size 40 \
                                --num_inner_updates 1 \
                                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta

            done
        done
    done
done