#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=16000
#SBATCH --time=24:00:00
#SBATCH --parsable

# Testing MAML with params from the paper

echo 'MAML started'

EPOCHS=1


for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done

for benchmark in Sinusoid1D
do
    for k_shot in 10 5
    do
        for num_inner_updates in 1
        do
            for inner_lr in 0.01 0.001

            do
                let num_points = k_shot * 2
                python train.py --algorithm maml \
                                --wandb True \
                                --nlml_testing_enabled True \
                                --num_epochs $EPOCHS \
                                --epochs_to_save 1 \
                                --num_episodes_per_epoch 20000 \
                                --benchmark $benchmark \
                                --num_models 1 \
                                --k_shot $k_shot \
                                --num_points_per_train_tasks $num_points \
                                --inner_lr $inner_lr \
                                --meta_lr 0.001 \
                                --minibatch 25 \
                                --noise_stddev 0.1 \
                                --num_hidden 2 \
                                --hidden_size 40 \
                                --num_episodes 4 \
                                --num_inner_updates $num_inner_updates \
                                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta
            done
        done
    done
done