#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=16000
#SBATCH --time=8:00:00
#SBATCH --parsable

echo 'Job started'

EPOCHS=60000
EPOCHS_TO_STORE=5000

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    export "$KEY"=$VALUE
done

echo $BENCHMARK

for num_samples in 1 2 4 8 1000
do
        for seed in 1234 4321 9999
        do

		python train.py --algorithm $ALGORITHM \
		                --wandb True \
		                --num_epochs $EPOCHS \
		                --benchmark $BENCHMARK \
		                --num_models $NUM_MODELS \
		                --k_shot $num_samples \
		                --num_episodes_per_epoch $EPOCHS_TO_STORE \
		                --seed $seed \
		                --seed_offset $seed \
		                --seed_offset_test $seed \
				        --logdir_base /pfs/work7/workspace/scratch/utpqw-meta
	done
done