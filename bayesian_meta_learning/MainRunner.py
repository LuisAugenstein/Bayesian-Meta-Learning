import os
from bayesian_meta_learning import visualizer
from bayesian_meta_learning.model_tester import calculate_loss_metrics
from bayesian_meta_learning.benchmark.benchmark_dataloader import create_benchmark_dataloaders
import wandb
import numpy as np
import torch
import random
from bayesian_meta_learning.algorithms.Baseline import Baseline
from few_shot_meta_learning.Maml import Maml
from few_shot_meta_learning.Platipus import Platipus
from few_shot_meta_learning.Bmaml import Bmaml


class MainRunner():
    def __init__(self, config) -> None:
        self.config = config
        # set seeds
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        # initialize algorithm to use
        algorithms = {
            'maml': Maml,
            'bmaml': Bmaml,
            'platipus': Platipus,
            'baseline': Baseline
        }
        self.algo = algorithms[config['algorithm']](config)
        if self.config['wandb']:
            wandb.define_metric(name="evaluation/run_id")
            wandb.define_metric(name="evaluation/*",
                                step_metric="evaluation/run_id")
        return

    def run(self) -> None:
        # create dataloaders
        train_dataloader, val_dataloader, test_dataloader = create_benchmark_dataloaders(
            self.config)

        checkpoint_path = os.path.join(
            self.config['logdir'], f"Epoch_{self.config['num_models']}.pt")

        # Only train if retraining is requested or no model with the current config exists
        if not self.config['reuse_models'] or not os.path.exists(checkpoint_path):
            self.algo.train(train_dataloader, val_dataloader)

        # set seeds again after training
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])
        # NLML calculation on test set
        if self.config['nlml_testing_enabled']:
            print("Testing is started.")
            if self.config['algorithm'] == 'platipus':
                self._test_platipus(test_dataloader)
            else:
                nlml, mse = calculate_loss_metrics(
                    self.algo, test_dataloader, self.config)
                if(self.config['wandb']):
                    wandb.log({
                        'evaluation/run_id': 1,
                        'evaluation/nlml': nlml,
                        'evaluation/mse': mse
                    })
                print(f"Test-NLML: {nlml}")
                print(f"Test-MSE: {mse}")

        # # visualize the dataset (training, validation and testing)
        print("Visualization is started.")
        print(f"Plots are stored at {self.config['logdir_plots']} \n")
        visualizer.plot_tasks_initially(
            'Meta_Training_Tasks', self.algo, train_dataloader, self.config)
        visualizer.plot_tasks_initially(
            'Meta_Validation_Tasks', self.algo, val_dataloader, self.config)
        visualizer.plot_tasks_initially(
            'Meta_Testing_Tasks', self.algo, test_dataloader, self.config)
        # # visualize predictions for training and validation tasks of each stored model
        for epoch in range(self.config['epochs_to_save'], self.config['num_epochs']+1, self.config['epochs_to_save']):
            visualizer.plot_task_results(
                'Training', epoch, self.algo, train_dataloader, self.config)
            visualizer.plot_task_results(
                'Validation', epoch, self.algo, val_dataloader, self.config)
        # visualize Predictions for test tasks of only the last model
        visualizer.plot_task_results(
            'Testing', self.config['num_epochs'], self.algo, test_dataloader, self.config)

    def _test_platipus(self, test_dataloader):
        old_num_models = self.config['num_models']
        num_runs = 100
        num_models_array = [5, 10, 20, 50, 100]
        nlmls = np.zeros((len(num_models_array), num_runs))
        mses = np.zeros((len(num_models_array), num_runs))
        print('{:<10} {:<10} {:<15} {:<15}'.format(
            'num_models', 'run_id', 'NLML', 'MSE'))
        for model_id, num_models in enumerate(num_models_array):
            for run_id in range(num_runs):
                self.algo.config['num_models'] = num_models
                self.config['num_models'] = num_models
                nlml, mse = calculate_loss_metrics(
                    self.algo, test_dataloader, self.config)
                if(self.config['wandb']):
                    wandb.log({
                        'evaluation/run_id': run_id,
                        f'evaluation/nlml_num_models_{num_models}': nlml,
                        f'evaluation/mse_num_models_{num_models}': mse
                    })
                print('{:<10} {:<10} {:<15} {:<15}'.format(num_models, run_id+1,
                      np.round(nlml, 4), np.round(mse, 4)))
                nlmls[model_id, run_id] = nlml
                mses[model_id, run_id] = mse
        print('\nAverage and std_dev over all runs')
        print('{:<10} {:<30} {:<30}'.format('num_models', 'NLML', 'MSE'))
        for model_id, num_models in enumerate(num_models_array):
            nlml = np.round(np.mean(nlmls[model_id]), 4)
            nlml_std = np.round(np.std(nlmls[model_id]), 4)
            mse = np.round(np.mean(mses[model_id]), 4)
            mse_std = np.round(np.std(mses[model_id]), 4)
            nlml_string = f'{nlml} +- {nlml_std}'
            mse_string = f'{mse} +- {mse_std}'
            print('{:<10} {:<30} {:<30}'.format(num_models, nlml_string, mse_string))
        print("")
        # restore num_models
        self.algo.config['num_models'] = old_num_models
        self.config['num_models'] = old_num_models
