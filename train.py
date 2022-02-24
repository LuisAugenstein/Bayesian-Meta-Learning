import torch
import numpy as np
import os
import argparse
import pathlib
import uuid

from bayesian_meta_learning.runner.MainRunner import MainRunner
from bayesian_meta_learning.runner.CLVRunner import CLVRunner
from few_shot_meta_learning._utils import train_val_split_regression

# --------------------------------------------------
# SETUP INPUT PARSER
# --------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    # Own arguments
    parser.add_argument("--load_dir_bmaml_chaser", default='bmaml_chaser_models/model_100.pickle', type=str,
                        help='path to a .pickle file with parameters from bmaml with chaser loss')
    parser.add_argument("--noise_stddev", default=0.1, type=float,
                        help='standard deviation of the white gaussian noise added to the data targets y')
    parser.add_argument("--seed", default=123, type=int,
                        help='general seed for everything but data generation')
    parser.add_argument("--seed_offset", default=1234, type=int,
                        help='data generation seed for the meta training tasks')
    parser.add_argument("--seed_offset_test", default=12345, type=int,
                        help='data generation seed for the meta testing task')

    parser.add_argument("--num_points_per_train_task", default=512, type=int,
                        help='number of datapoints in each meta training task')
    parser.add_argument("--num_test_tasks", default=100, type=int,
                        help='number of meta testing tasks')
    parser.add_argument("--num_points_per_test_task", default=512, type=int,
                        help='number of datapoints in each meta testing task')

    parser.add_argument("--num_hidden", default=2, type=int,
                        help='number of hidden layers if using a fully connceted network')
    parser.add_argument("--hidden_size", default=40, type=int,
                        help='hidden layer size if using a fully connected network')

    parser.add_argument("--advance_leader", default=10, type=int,
                        help='number of training steps the leader is ahead of the chaser(s). Only relevant for bmaml-chaser')

    parser.add_argument("--num_inner_updates_testing", default=5, type=int,
                        help="number of inner update steps used during testing")
    parser.add_argument("--nlml_testing_enabled", default=False, type=bool,
                        help="whether to calculate neg log marginal likelihood or not.")
    parser.add_argument("--reuse_models", default=False, type=bool,
                        help='Specifies if a saved state should be used if found or if the model should be trained from start.')
    parser.add_argument("--normalize_benchmark", default=False, type=bool)

    parser.add_argument("--wandb", default=False, type=bool,
                        help="Specifies if logs should be written to WandB")
    parser.add_argument("--num_visualization_tasks", default=6, type=int,
                        help='number of randomly chosen meta testing tasks that are used for visualization')
    parser.add_argument("--y_plotting_resolution", default=512, type=int,
                        help="number of discrete y-axis points to evaluate for visualization")
    parser.add_argument("--epochs_to_save", default=1000, type=int,
                        help="number of epochs between saving the model")

    # fsml arguments
    parser.add_argument("--benchmark", default='Sinusoid1D',
                        help='possible values are Sinusoid1D, Affine1D, Quadratic1D, SinusoidAffine1D')
    parser.add_argument("--num_ways", default=1, type=int,
                        help='d_y dimension of targets')
    parser.add_argument("--k_shot", default=5, type=int,
                        help='number of datapoints in the context set (needs to be less than points_per_train_task)')

    parser.add_argument("--algorithm", default='maml',
                        help='possible values are maml, platipus, bmaml, baseline and bmaml_chaser and clv')
    parser.add_argument("--network_architecture", default="FcNet")
    parser.add_argument("--num_epochs", default=5, type=int,
                        help='number of training epochs. one epoch corresponds to one meta update for theta. model is stored all 500 epochs')
    parser.add_argument('--num_episodes_per_epoch', default=20, type=int,
                        help='Number of meta train tasks. should be a multiple of minibatch')
    parser.add_argument("--num_models", default=10, type=int,
                        help='number of models (phi) we sample from the posterior in the end for evaluation. irrelevant for maml')
    parser.add_argument('--minibatch', default=25, type=int,
                        help='Minibatch of episodes (tasks) to update meta-parameters')
    parser.add_argument('--minibatch_print', default=1, type=int,
                        help='number of minibatches between each validation plotting to wandb')
    parser.add_argument("--num_inner_updates", default=1, type=int,
                        help='number of SGD steps during adaptation')
    parser.add_argument("--inner_lr", default=0.001, type=float)
    parser.add_argument("--meta_lr", default=0.001, type=float)
    parser.add_argument("--KL_weight", default=0.01, type=float)
    parser.add_argument('--num_episodes', type=int, default=4,
                        help='Number of validation tasks used for the MLBaseClass.evaluate() method')
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

    # check if minibatch is valid
    if config['minibatch'] > config['num_episodes_per_epoch']:
        print(f'invalid config: \n' +
              f'minibatch={config["minibatch"]} needs to be smaller than num_episodes_per_epoch={config["num_episodes_per_epoch"]}. \n' +
              f'new value minibatch={config["num_episodes_per_epoch"]}. \n')
        config['minibatch'] = config['num_episodes_per_epoch']

    # check if minibatch_print is valid
    config['minibatch_print'] *= config['minibatch']
    if config['minibatch_print'] > config['num_episodes_per_epoch']:
        print(f'invalid config: \n' +
              f'minibatch_print={config["minibatch"]} needs to be smaller than num_episodes_per_epoch={config["num_episodes_per_epoch"]}. \n' +
              f'new value minibatch_print={config["num_episodes_per_epoch"]}. \n')
        config['minibatch_print'] = config['num_episodes_per_epoch']

    # check if epochs_to_save is valid
    if config['epochs_to_save'] > config['num_epochs']:
        print(f'invalid config: \n' +
              f'epochs_to_save={config["epochs_to_save"]} needs to be smaller or equal than num_epochs={config["num_epochs"]}. \n' +
              f'new value epochs_to_save={config["num_epochs"]}. \n')
        config['epochs_to_save'] = config['num_epochs']
    
    if config['num_test_tasks'] % config['minibatch'] != 0:
            print("!!!!!WARNING!!!!! num_test_tasks should be a multiple of minibatch.")
            print(f"new Value for num_test_tasks is {config['minibatch']}")
            config['num_test_tasks'] = config['minibatch']

    # create directory tree to store models and plots
    # If wandb is disabled, save the config file into the directory
    create_save_models_directory(config)

    config['loss_function'] = torch.nn.MSELoss()
    config['train_val_split_function'] = train_val_split_regression

    config['device'] = torch.device('cuda:0') if torch.cuda.is_available() \
        else torch.device('cpu')

    # start the run
    if config['algorithm'] == 'clv':
        runner = CLVRunner(config)
    else: 
        runner = MainRunner(config)
    runner.run()


def create_save_models_directory(config: dict):
    if config['wandb']:
        identifier = uuid.uuid1()
        logdir = os.path.join(
            config['logdir_base'], 'saved_models', str(identifier))

        config['logdir'] = logdir
        config['logdir_plots'] = logdir
        pathlib.Path(config['logdir']).mkdir(parents=True, exist_ok=True)
    else:
        logdir = os.path.join(config['logdir_base'], 'saved_models',
                              config['algorithm'].lower(),
                              config['network_architecture'],
                              config['benchmark'],
                              f"{config['k_shot']}-shot",
                              f"{config['num_models']}-models",
                              )
        config['logdir_plots'] = os.path.join(logdir, 'plots')
        pathlib.Path(config['logdir_plots']).mkdir(parents=True, exist_ok=True)

        config['logdir'] = os.path.join(logdir, 'models')
        pathlib.Path(config['logdir']).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
