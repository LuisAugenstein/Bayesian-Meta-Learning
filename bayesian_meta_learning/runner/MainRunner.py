from copy import deepcopy
import torch
import os
import numpy as np
import random

from bayesian_meta_learning import visualizer
from bayesian_meta_learning.benchmark.benchmark_dataloader import create_benchmark_dataloaders
from bayesian_meta_learning.algorithms import Baseline
from bayesian_meta_learning.nlml_tester import test_neg_log_marginal_likelihood

from few_shot_meta_learning.Maml import Maml
from few_shot_meta_learning.Platipus import Platipus
from few_shot_meta_learning.Bmaml import Bmaml
from few_shot_meta_learning.HyperNetClasses import IdentityNet, NormalVariationalNet


def apply_random_seed(num: int):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)


class MainRunner():
    def __init__(self, config) -> None:
        self.config = config

        apply_random_seed(config['seed'])

        self.train_dataloader, self.val_dataloader, self.test_dataloader = create_benchmark_dataloaders(
            config)

        algorithms = {
            'maml': Maml,
            'bmaml': Bmaml,
            'platipus': Platipus,
            'baseline': Baseline
        }
        self.algo = algorithms[config['algorithm']](config)

    def run(self) -> None:
        checkpoint_path = os.path.join(
            self.config['logdir'], f"Epoch_{self.config['num_models']}.pt")

        # Only train if retraining is requested or no model with the current config exists
        if not self.config['reuse_models'] or not os.path.exists(checkpoint_path):
            self.algo.train(train_dataloader=self.train_dataloader,
                            val_dataloader=self.val_dataloader)
        # test
        test_neg_log_marginal_likelihood(self.algo, self.test_dataloader, self.config)

        # # visualize the dataset (training, validation and testing)
        print("Visualization is started.")
        print(f"Plots are stored at {self.config['logdir_plots']}")
        visualizer.plot_tasks_initially(
            'Meta_Training_Tasks', self.algo, self.train_dataloader, self.config)
        visualizer.plot_tasks_initially(
            'Meta_Validation_Tasks', self.algo, self.val_dataloader, self.config)
        visualizer.plot_tasks_initially(
            'Meta_Testing_Tasks', self.algo, self.test_dataloader, self.config)
        # # visualize predictions for training and validation tasks of each stored model
        for epoch in range(1, self.config['num_epochs']+1):
            visualizer.plot_task_results(
                'Training', epoch, self.algo, self.train_dataloader, self.config)
            visualizer.plot_task_results(
                'Validation', epoch, self.algo, self.val_dataloader, self.config)
        # visualize Predictions for test tasks of only the last model
        visualizer.plot_task_results(
            'Testing', self.config['num_epochs'], self.algo, self.test_dataloader, self.config)
        # # TODO: Calculate/Query all the statistics we want to know about...
