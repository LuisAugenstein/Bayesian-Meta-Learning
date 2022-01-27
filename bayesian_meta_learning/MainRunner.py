import os
from bayesian_meta_learning import visualizer
from bayesian_meta_learning.nlml_tester import test_neg_log_marginal_likelihood
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
            wandb.define_metric(name="nlml_test/run_id")
            wandb.define_metric(name="nlml_test/*",
                                step_metric="nlml_test/run_id")
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
                nlml = test_neg_log_marginal_likelihood(
                    self.algo, test_dataloader, self.config)
                if(self.config['wandb']):
                    wandb.log({
                        'nlml_test/run_id': 1,
                        'nlml_test/loss': nlml
                    })
                print(f"Test-NLML: {nlml} \n")

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
        means = []
        stds = []
        num_models_array = [5, 10, 20, 50, 100]
        for num_models in num_models_array:
            nlml = [float] * num_runs
            for run_id in range(num_runs):
                self.algo.config['num_models'] = num_models
                self.config['num_models'] = num_models
                nlml[run_id] = test_neg_log_marginal_likelihood(
                    self.algo, test_dataloader, self.config)
                if(self.config['wandb']):
                    wandb.log({
                        'nlml_test/run_id': run_id,
                        f'nlml_test/loss_num_models_{num_models}': nlml[run_id]
                    })
                print(f"{num_models}_{run_id+1}: {np.round(nlml[run_id], 4)}")
            means.append(np.round(np.mean(nlml), 4))
            stds.append(np.round(np.std(nlml), 4))
        for i, num_models in enumerate(num_models_array):
            print(f"Test-NLML_{num_models}: {means[i]} +- {stds[i]}")
        print("")
        # restore num_models
        self.algo.config['num_models'] = old_num_models
        self.config['num_models'] = old_num_models
