#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=32000
#SBATCH --time=24:00:00
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


python -W ignore train.py 	--algorithm $algorithm \
							--nlml_testing_enabled True \
							--wandb True \
							--logdir_base /pfs/work7/workspace/scratch/utpqw-meta \
							--seed 9999 \
							--num_epochs $num_epochs \
							--num_episodes_per_epoch $num_episodes_per_epoch \
							--epochs_to_save $epochs_to_save \
							--benchmark $benchmark \
							--num_models $num_models \
							--k_shot $k_shot \
							--num_points_per_train_task $num_points_per_train_task \
							--inner_lr $inner_lr \
							--meta_lr $meta_lr \
							--minibatch $minibatch \
							--noise_stddev $noise_stddev \
							--num_hidden $num_hidden \
							--hidden_size $hidden_size \
							--num_episodes $num_episodes \
							--num_inner_updates $num_inner_updates \
							--KL_weight $KL_weight
