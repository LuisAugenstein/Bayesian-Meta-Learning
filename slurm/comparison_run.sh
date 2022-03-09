#!/bin/bash

# Train all algorithms on the same data.
# Reun the algortihms for on 10 different seed.
# Algorithms are trained in parallel in different batch jobs.

num_episodes_per_epoch=600000

for seed in 1 2 3 4 5 6 7 8 9 10:
    for benchmark in Sinusoid1D Quadratic1D SinusoidAffine1D:

        sbatch slurm/start_job.sh   algorithm=maml \
                                    seed=$seed \
                                    num_epochs=10 \
                                    num_episodes_per_epoch=$num_episodes_per_epoch \
                                    epochs_to_save=1 \
                                    benchmark=$benchmark \
                                    num_models=1 \
                                    k_shot=$k_shot \
                                    num_points_per_train_task=50 \
                                    inner_lr=0.001 \
                                    meta_lr=0.001 \
                                    minibatch=25 \
                                    noise_stddev=0.1 \
                                    num_hidden=3 \
                                    hidden_size=40 \
                                    num_episodes=4 \
                                    num_inner_updates=5 \
                                    KL_weight=0

        sbatch slurm/start_job.sh   algorithm=platipus \
                                    seed=$seed \
                                    num_epochs=10 \
                                    num_episodes_per_epoch=$num_episodes_per_epoch \
                                    epochs_to_save=1 \
                                    benchmark=$benchmark \
                                    num_models=10 \
                                    k_shot=$k_shot \
                                    num_points_per_train_task=50 \
                                    inner_lr=0.001 \
                                    meta_lr=0.001 \
                                    minibatch=25 \
                                    noise_stddev=0.1 \
                                    num_hidden=3 \
                                    hidden_size=40 \
                                    num_episodes=4 \
                                    num_inner_updates=5 \
                                    KL_weight=0.01

        sbatch slurm/start_job.sh   algorithm=bmaml \
                                    seed=$seed \
                                    num_epochs=10 \
                                    num_episodes_per_epoch=$num_episodes_per_epoch \
                                    epochs_to_save=1 \
                                    benchmark=$benchmark \
                                    num_models=10 \
                                    k_shot=$k_shot \
                                    num_points_per_train_task=50 \
                                    inner_lr=0.001 \
                                    meta_lr=0.001 \
                                    minibatch=25 \
                                    noise_stddev=0.1 \
                                    num_hidden=3 \
                                    hidden_size=40 \
                                    num_episodes=4 \
                                    num_inner_updates=5 \
                                    KL_weight=0

        sbatch slurm/start_job.sh   algorithm=baseline \
                                    seed=$seed \
                                    num_epochs=10 \
                                    num_episodes_per_epoch=$num_episodes_per_epoch \
                                    epochs_to_save=1 \
                                    benchmark=$benchmark \
                                    num_models=1 \
                                    k_shot=$k_shot \
                                    num_points_per_train_task=50 \
                                    inner_lr=0.001 \
                                    meta_lr=0.001 \
                                    minibatch=25 \
                                    noise_stddev=0.1 \
                                    num_hidden=3 \
                                    hidden_size=40 \
                                    num_episodes=4 \
                                    num_inner_updates=5 \
                                    KL_weight=0

        sbatch slurm/start_job.sh   algorithm=clv \
                                    seed=$seed \
                                    num_epochs=10 \
                                    num_episodes_per_epoch=$num_episodes_per_epoch \
                                    epochs_to_save=0 \
                                    benchmark=$benchmark \
                                    num_models=10 \
                                    k_shot=$k_shot \
                                    num_points_per_train_task=50 \
                                    inner_lr=0 \
                                    meta_lr=0 \
                                    minibatch=25 \
                                    noise_stddev=0.1 \
                                    num_hidden=3 \
                                    hidden_size=40 \
                                    num_episodes=4 \
                                    num_inner_updates=0 \
                                    KL_weight=0
        
    done
done


