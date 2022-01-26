import os
from bayesian_meta_learning import visualizer
from bayesian_meta_learning.nlml_tester import test_neg_log_marginal_likelihood
from bayesian_meta_learning.benchmark.benchmark_dataloader import create_benchmark_dataloaders
from bayesian_meta_learning.runner.BaseRunner import BaseRunner


class MainRunner(BaseRunner):
    def __init__(self, config) -> None:
        super().__init__(config=config)

    def run(self) -> None:
        # create dataloaders
        train_dataloader, val_dataloader, test_dataloader = create_benchmark_dataloaders(
            self.config)

        checkpoint_path = os.path.join(
            self.config['logdir'], f"Epoch_{self.config['num_models']}.pt")

        # Only train if retraining is requested or no model with the current config exists
        if not self.config['reuse_models'] or not os.path.exists(checkpoint_path):
            self.algo.train(train_dataloader, val_dataloader)
        # test
        test_neg_log_marginal_likelihood(self.algo, test_dataloader, self.config)

        # # visualize the dataset (training, validation and testing)
        print("Visualization is started.")
        print(f"Plots are stored at {self.config['logdir_plots']}")
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