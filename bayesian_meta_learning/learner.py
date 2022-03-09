import os
import pathlib
import uuid
import torch
from bayesian_meta_learning.parameter_description import parameter_description
from bayesian_meta_learning.runner.MainRunner import MainRunner
from bayesian_meta_learning.runner.CLVRunner import CLVRunner
from few_shot_meta_learning._utils import train_val_split_regression

print("Hallo")


class Learner():

    def get_default_config():
        config = {}
        for entry in parameter_description:
            config[entry['name']] = entry['default']
        return config

    def run(config: dict):
        Learner.check_config_validity(config)
        Learner.create_save_models_directory(config)

        config['loss_function'] = torch.nn.MSELoss()
        config['train_val_split_function'] = train_val_split_regression
        config['device'] = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu')

        if config['algorithm'] == 'clv':
            runner = CLVRunner(config)
        else: 
            runner = MainRunner(config)
        runner.run()

    def check_config_validity(config: dict):
        if config['minibatch'] > config['num_episodes_per_epoch']:
            print(f'invalid config: \n' +
                  f'minibatch={config["minibatch"]} needs to be smaller than num_episodes_per_epoch={config["num_episodes_per_epoch"]}. \n' +
                  f'new value minibatch={config["num_episodes_per_epoch"]}. \n')
            config['minibatch'] = config['num_episodes_per_epoch']

        config['minibatch_print'] *= config['minibatch']
        if config['minibatch_print'] > config['num_episodes_per_epoch']:
            print(f'invalid config: \n' +
                  f'minibatch_print={config["minibatch"]} needs to be smaller than num_episodes_per_epoch={config["num_episodes_per_epoch"]}. \n' +
                  f'new value minibatch_print={config["num_episodes_per_epoch"]}. \n')
            config['minibatch_print'] = config['num_episodes_per_epoch']

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
                                  f"{config['num_models']}-models",
                                  )
            config['logdir_plots'] = os.path.join(logdir, 'plots')
            pathlib.Path(config['logdir_plots']).mkdir(
                parents=True, exist_ok=True)

            config['logdir'] = os.path.join(logdir, 'models')
            pathlib.Path(config['logdir']).mkdir(parents=True, exist_ok=True)
