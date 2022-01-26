#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=16000
#SBATCH --time=8:00:00
#SBATCH --parsable

for algo in maml bmaml platipus
do
    python train.py --algorithm $algorithm \
                    --wandb True \
                    --num_epochs 1 \
                    --benchmark Sinusoid1D \
                    --num_models 10 \
                    --k_shot 100 \
                    --minibatch 20 \
                    --num_episodes_per_epoch 20 \
                    --noise_stddev 0.02 \
                    --num_inner_updates 10000 \
                    --logdir_base /pfs/work7/workspace/scratch/utpqw-meta
                    
    python train.py --algorithm $algorithm \
                    --wandb True \
                    --num_epochs 10000 \
                    --benchmark Sinusoid1D \
                    --num_models 10 \
                    --k_shot 5 \
                    --minibatch 1 \
                    --num_episodes_per_epoch 1 \
                    --num_validation_tasks 0 \
                    --noise_stddev 0.02 \
                    --num_inner_updates 1 \
                    --logdir_base /pfs/work7/workspace/scratch/utpqw-meta
done