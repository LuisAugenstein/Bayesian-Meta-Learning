import torch
import numpy as np
import os
import argparse
import pathlib
from bayesian_meta_learning.runner.MainRunner import MainRunner
from bayesian_meta_learning.runner.NlmlRunner import NlmlRunner
from few_shot_meta_learning._utils import train_val_split_regression

# --------------------------------------------------
# SETUP INPUT PARSER
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    # Own arguments
    parser.add_argument("--runner", default='main_runner', type=str,
                        help='main_runner, nlml_runner')
    parser.add_argument("--noise_stddev", default=0.02, type=float,
                        help='standard deviation of the white gaussian noise added to the data targets y')
    parser.add_argument("--seed", default=123, type=int,
                        help='general seed for everything but data generation')
    parser.add_argument("--seed_offset", default=1234, type=int,
                        help='data generation seed for the meta training tasks')
    parser.add_argument("--seed_offset_test", default=12345, type=int,
                        help='data generation seed for the meta testing task')

    parser.add_argument("--num_train_tasks", default=16, type=int,
                        help='number of meta training tasks')
    parser.add_argument("--num_points_per_train_task", default=512, type=int,
                        help='number of datapoints in each meta training task')
    parser.add_argument("--num_validation_tasks", default=4, type=int,
                        help='number of tasks used for validation during training')
    parser.add_argument("--num_test_tasks", default=20, type=int,
                        help='number of meta testing tasks')
    parser.add_argument("--num_points_per_test_task", default=512, type=int,
                        help='number of datapoints in each meta testing task')

    parser.add_argument("--reuse_models", default=False, type=bool,
                        help='Specifies if a saved state should be used if found or if the model should be trained from start.')
    parser.add_argument("--normalize_benchmark", default=True, type=bool)

    parser.add_argument("--wandb", default=False, type=bool,
                        help="Specifies if logs should be written to WandB")
    parser.add_argument("--num_visualization_tasks", default=4, type=int,
                        help='number of randomly chosen meta testing tasks that are used for visualization')
    parser.add_argument("--y_plotting_resolution", default=512, type=int,
                        help="number of discrete y-axis points to evaluate for visualization")

    # fsml arguments
    parser.add_argument("--benchmark", default='Sinusoid1D',
                        help='possible values are Sinusoid1D, Affine1D, Quadratic1D, SinusoidAffine1D')
    parser.add_argument("--num_ways", default=1, type=int,
                        help='d_y dimension of targets')
    parser.add_argument("--k_shot", default=1, type=int,
                        help='number of datapoints in the context set (needs to be less than points_per_train_task)')

    parser.add_argument("--algorithm", default='maml',
                        help='possible values are maml, platipus, bmaml and baseline')
    parser.add_argument("--network_architecture", default="FcNet")
    parser.add_argument("--num_epochs", default=5, type=int,
                        help='number of training epochs. one epoch corresponds to one meta update for theta. model is stored all 500 epochs')
    parser.add_argument('--num_episodes_per_epoch', default=10000, type=int,
                        help='Save meta-parameters after this number of episodes')
    parser.add_argument("--num_models", default=5, type=int,
                        help='number of models (phi) we sample from the posterior in the end for evaluation. irrelevant for maml')
    parser.add_argument('--minibatch', default=20, type=int,
                        help='Minibatch of episodes (tasks) to update meta-parameters')
    parser.add_argument('--minibatch_print', default=10, type=int,
                        help='number of minibatches between each validation printing')
    parser.add_argument("--num_inner_updates", default=5, type=int,
                        help='number of SGD steps during adaptation')
    parser.add_argument("--inner_lr", default=0.01, type=float)
    parser.add_argument("--meta_lr", default=1e-3, type=float)
    parser.add_argument("--KL_weight", default=1e-6, type=float)
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes used in testing')
    parser.add_argument("--resume_epoch", default=0,
                        help='0 means fresh training. >0 means training continues from a corresponding stored model.')
    parser.add_argument('--logdir_base', default='.', type=str,
                        help='Folder to store model and logs')
    parser.add_argument("--first_order", default=True, type=bool,
                        help="Should always be true for MAML basd algos")
    parser.add_argument("--train_flag", default=True, type=bool)

    args = parser.parse_args()

    # define config object from parser args
    config = {}
    for key in args.__dict__:
        config[key] = args.__dict__[key]

    config['minibatch_print'] *= config['minibatch']
    config['loss_function'] = torch.nn.MSELoss()
    config['train_val_split_function'] = train_val_split_regression

    config['device'] = torch.device('cuda:0') if torch.cuda.is_available() \
        else torch.device('cpu')

    # create directory tree to store models and plots
    create_save_models_directory(config) 

    # choose a Runner and start the run
    runners = {
        'main_runner': MainRunner,
        'nlml_runner': NlmlRunner
    }
    runner = runners[config['runner']](config)
    runner.run()

def create_save_models_directory(config: dict):
    logdir = os.path.join(config['logdir_base'], 'saved_models',
                          config['algorithm'].lower(),
                          config['network_architecture'],
                          config['benchmark'],
                          f"{config['k_shot']}-shot",
                          f"{config['num_models']}-models",
                          f"{config['seed']}_{config['seed_offset']}_{config['seed_offset_test']}"
                          )

    config['logdir'] = os.path.join(logdir, 'models')
    config['logdir_plots'] = os.path.join(logdir, 'plots')
    pathlib.Path(config['logdir']).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config['logdir_plots']).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
