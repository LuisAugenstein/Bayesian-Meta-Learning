#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=64000
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

for benchmark in Sinusoid1D
do
    for k_shot in 5
    do
        for num_inner_updates in 5
        do
            for noise_std_dev in 0.3 0.1
            do
                for num_models in 4 10
                do
                    for kl_weight in 1.5 0.15 0.01 0.0001
                    do
            			let num_points=$((k_shot * 2))
                        sbatch slurm/start_job.sh     --algorithm=platipus \
                                                    --num_epochs=$EPOCHS \
                                                    --num_episodes_per_epoch=60000 \
                                                    --epochs_to_save=1 \
                                                    --benchmark=$benchmark \
                                                    --num_models=10 \
                                                    --k_shot=$k_shot \
                                                    --num_points_per_train_task=$num_points \
                                                    --inner_lr=0.01 \
                                                    --meta_lr=0.001 \
                                                    --minibatch 25 \
                                                    --noise_stddev=$noise_std_dev \
                                                    --num_hidden=3 \
                                                    --hidden_size=100 \
                                                    --num_episodes=4 \
                                                    --num_inner_updates=$num_inner_updates \
                                                    --KL_weight=$kl_weight \
                    done
                done
            done
        done
    done
done
