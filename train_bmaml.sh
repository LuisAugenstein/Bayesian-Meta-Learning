#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=single
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --parsable

num_epochs=10000 
num_episodes_per_epoch=100 
epochs_to_save=1 
benchmark=Sinusoid1D
num_models=1 
k_shot=5 
num_points_per_train_task=50 
inner_lr=0.001 
meta_lr=0.001 
minibatch=10
noise_stddev=0.1 
num_hidden=2 
hidden_size=40 
num_episodes=4 
num_inner_updates=10 
seed=123

eval "$(conda shell.bash hook)"
conda activate bmaml

cd bmaml

python bmaml_main.py \
    --finite=True \
    --train_total_num_tasks=$num_episodes_per_epoch \
    --test_total_num_tasks=$100 \
    --num_particles=$num_models \
    --num_tasks=$minibatch \
    --few_k_shot=$k_shot \
    --val_k_shot=$k_shot \
    --num_epochs=$num_epochs \
    --meta_lr=$meta_lr \
    --dim_hidden=$hidden_size \
    --num_layers=$num_hidden \
    --seed=$seed

cd ..

eval "$(conda shell.bash hook)"
conda activate meta

python train.py --algorithm bmaml \
                --nlml_testing_enabled True \
                --wandb True \
                --logdir_base /pfs/work7/workspace/scratch/utpqw-meta \
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
                --KL_weight 0 \
                --load_dir_bmaml_chaser bmaml/model_weights.pickle