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
    for k_shot in 5
    do
        for particles in 10
        do
            for inner_lr in 0.01 0.001
            do
                let num_points = k_shot * 2

                python train.py --algorithm bmaml \
                                --wandb True \
                                --nlml_testing_enabled True \
                                --num_epochs 10000 \
                                --num_train_tasks 100 \
                                --benchmark $benchmark \
                                --num_models $particles \
                                --k_shot $k_shot \
                                --num_inner_updates 5 \
                                --num_episodes_per_epoch $EPOCHS_TO_STORE \
                                --num_points_per_train_tasks $num_points \
                                --inner_lr $inner_lr \
                                --meta_lr 0.001 \
                                --minibatch 10 \
                                --noise_stddev 0.05 \
                                --num_hidden 3 \
                                --hidden_size 40 \
                                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta

                python train.py --algorithm bmaml \
                                --wandb True \
                                --nlml_testing_enabled True \
                                --num_epochs 1000 \
                                --num_train_tasks 1000 \
                                --benchmark $benchmark \
                                --num_models $particles \
                                --k_shot $k_shot \
                                --num_inner_updates 5 \
                                --num_episodes_per_epoch $EPOCHS_TO_STORE \
                                --num_points_per_train_tasks $num_points \
                                --inner_lr $inner_lr \
                                --meta_lr 0.001 \
                                --minibatch 10 \
                                --noise_stddev 0.05 \
                                --num_hidden 3 \
                                --hidden_size 40 \
                                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta

            done
        done
    done
done