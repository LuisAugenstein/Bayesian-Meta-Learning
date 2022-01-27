#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=16000
#SBATCH --time=8:00:00
#SBATCH --parsable

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done


python train.py --algorithm $ALGO \
                --wandb True \
                --num_epochs 50000 \
                --benchmark $benchmark \
                --nlml_testing_enabled True \
                --num_models 10 \
                --k_shot 5 \
                --num_inner_updates 1 \
                --seed 5544 \
                --inner_lr 0.01 \
                --meta_lr 0.001 \
                --minibatch 25 \
                --num_episodes_per_epoch 25 \
                --noise_stddev 0.02 \
                --num_hidden 2 \
                --hidden_size 40 \
                --num_episodes 4 \
                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta