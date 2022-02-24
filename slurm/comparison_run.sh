#!/bin/bash

benchmark=Sinusoid1D
k_shot=5

sbatch slurm/start_job.sh   algorithm=maml \
				            num_epochs=10 \
				            num_episodes_per_epoch=60000 \
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


sbatch slurm/start_job.sh   algorithm=platipus \
                            num_epochs=10 \
                            num_episodes_per_epoch=60000 \
                            epochs_to_save=1 \
                            benchmark=$benchmark \
                            num_models=10 \
                            k_shot=$k_shot \
                            num_points_per_train_task=50 \
                            inner_lr=0.01 \
                            meta_lr=0.001 \
                            minibatch=25 \
                            noise_stddev=0.1 \
                            num_hidden=3 \
                            hidden_size=40 \
                            num_episodes=4 \
                            num_inner_updates=5 \
                            KL_weight=0.01

sbatch slurm/start_job.sh   algorithm=platipus \
                            num_epochs=10 \
                            num_episodes_per_epoch=60000 \
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
                            num_epochs=10 \
                            num_episodes_per_epoch=60000 \
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

sbatch slurm/start_job.sh   algorithm=baseline \
                            num_epochs=10 \
                            num_episodes_per_epoch=60000 \
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

sbatch slurm/start_job.sh   algorithm=clv \
                            num_epochs=10 \
                            num_episodes_per_epoch=60000 \
                            benchmark=$benchmark \
                            num_models=10 \
                            k_shot=$k_shot \
                            num_points_per_train_task=50 \
                            minibatch=25 \
                            noise_stddev=0.1 \


