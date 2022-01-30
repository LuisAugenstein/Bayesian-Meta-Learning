#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=64000
#SBATCH --time=24:00:00
#SBATCH --parsable

# Testing MAML with params from the paper

echo 'MAML started'

EPOCHS=100


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
		for episodes in 20000 50000 100000
		do
        	let num_points=$((k_shot * 2))
	        sbatch slurm/start_job.sh   algorithm=maml \
                                    num_epochs=$EPOCHS \
                                    num_episodes_per_epoch=$episodes \
                                    epochs_to_save=1 \
                                    benchmark=$benchmark \
                                    num_models=1 \
                                    k_shot=$k_shot \
                                    num_points_per_train_task=50 \
                                    inner_lr=$inner_lr \
                                    meta_lr=0.001 \
                                    minibatch=25 \
                                    noise_stddev=0.1 \
                                    num_hidden=2 \
                                    hidden_size=40 \
                                    num_episodes=4 \
                                    num_inner_updates=$num_inner_updates \
                                    KL_weight=1e-6
    		done
 	    done
        done
    done
done
