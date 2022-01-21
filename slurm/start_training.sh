#!/bin/bash

module load devel/cuda/11.0
module load devel/cudnn/10.2

for benchmark in Sinusoid1D Affine1D SinusoidAffine1D
do
	for num_models in 5 100 1000
	do
		sbatch start_job.sh ALGORITHM="platipus" NUM_MODELS=$num_models BENCHMARK=$benchmark
		sbatch start_job.sh ALGORITHM="bmaml" NUM_MODELS=$num_models BENCHMARK=$benchmark
	done

	sbatch start_job.sh ALGORITHM="maml" NUM_MODELS=1 BENCHMARK=$benchmark
done