from bayesian_meta_learning import visualizer
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningTask
from bayesian_meta_learning.model_tester import calculate_neg_log_marginal_likelihood
from bayesian_meta_learning.benchmark.benchmark_dataloader import create_benchmarks
import wandb
import numpy as np
import torch
import random

from metalearning_models import CLVModel


class CLVRunner():
    def __init__(self, config) -> None:
        self.config = config
        # set seeds
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        # setup wandb
        if (config['wandb']):
            wandb.init(project="fsml_" + config['algorithm'],
                       entity="seminar-meta-learning",
                       config=config)
            wandb.define_metric(name="meta_train/epoch")
            wandb.define_metric(name="meta_train/*",
                                step_metric="meta_train/epoch")

            wandb.define_metric(name="adapt/epoch")
            wandb.define_metric(name="adapt/*", step_metric="adapt/epoch")

            wandb.define_metric(name="results/sample")
            wandb.define_metric(name="results/*", step_metric="results/sample")
            
            wandb.define_metric(name="evaluation/run_id")
            wandb.define_metric(name="evaluation/*",
                                step_metric="evaluation/run_id")
        # initialize algorithm to use
        self.algo = CLVModel(config['logdir'])
        return

    def run(self) -> None:
        # create benchmarks
        bm_meta, bm_val, bm_test = create_benchmarks(self.config)

        all_model_specs = self.algo .get_all_model_specs()
        default_settings = self.algo.get_default_settings(
            d_x=1, d_y=1, d_param=5, n_context_meta=self.config['k_shot'], n_context_val=self.config['k_shot'], model_spec=all_model_specs[0])

        # update the settings to your needs
        settings = default_settings
        settings["d_z"] = 32
        settings["adam_lr"] = 0.01
        settings["f_act"] = "relu"
        settings["batch_size"] = self.config['minibatch']
        settings["loss_type"] = "MC"
        settings["decoder_kwargs"]["arch"] = "separate_networks"
        settings["loss_kwargs"]["n_marg"] = self.config['num_models']

        # construct the model architecture with the updated settings
        self.algo.initialize_new_model(seed=1234, settings=settings)

        # train and save the model
        self.algo.meta_train(benchmark_meta=bm_meta,
                             benchmark_val=bm_val,
                             max_tasks=self.config['num_epochs']*self.config['num_episodes_per_epoch'])
        self.algo.save_model()

        # set seeds again after training
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])

        # get test data
        x_test = torch.zeros(
            (bm_test.n_task, bm_test.n_datapoints_per_task, 1))
        y_test = torch.zeros(
            (bm_test.n_task, bm_test.n_datapoints_per_task, 1))
        for i in range(bm_test.n_task):
            task = bm_test.get_task_by_index(i)
            x = torch.tensor(task.x, dtype=torch.float32)
            y = torch.tensor(task.y, dtype=torch.float32)
            x_test[i], sort_indices = torch.sort(x, dim=0)
            y_test[i] = y[sort_indices.squeeze()]

        # get context data
        ids = [i for i in range(x_test.shape[1])]
        x_ctx = torch.zeros(self.config['num_test_tasks'], self.config['k_shot'], 1)
        y_ctx = torch.zeros(self.config['num_test_tasks'], self.config['k_shot'], 1)
        for i in range(self.config['num_test_tasks']):
            k_ids=random.sample(population = ids, k = self.config['k_shot'])
            x_ctx[i] = x_test[i, k_ids]
            y_ctx[i] = y_test[i, k_ids]

        # adapt on context data and predict test data
        y_pred = self.predict(x_ctx, y_ctx, x_test)

        # reshape to [n_tasks, n_samples, n_datapoints_per_task]
        y_pred = y_pred.squeeze()
        y_test = y_test.reshape((bm_test.n_task, 1, bm_test.n_datapoints_per_task))

        # compare different samples from task 0 predictions
        y_predictions = [y_pred[0, 0, :], y_pred[0, 1, :], y_pred[0, 2, :], y_pred[0, 3, :]]

        # calculate losses
        self.calculate_losses(y_pred, y_test)

        # visualize the test data
        print("Visualization is started.")
        print(f"Plots are stored at {self.config['logdir_plots']} \n")
        num_visualization_tasks = np.min(
            [self.config['num_visualization_tasks'], bm_test.n_task])
        # the None is because we have only 1 sample
        y_test = y_test.squeeze()[:, None, :]
        x_test = x_test.squeeze()[:, None, :]
        y_ctx = y_ctx.squeeze()[:, None, :]
        x_ctx = x_ctx.squeeze()[:, None, :]
        plotting_data = visualizer.generate_plotting_data(
            y_pred, y_test, x_test, y_ctx, x_ctx, num_visualization_tasks, self.config)
        visualizer.generate_plots(
            'Testing', self.config['num_epochs'], plotting_data, self.config)


    # x_ctx, y_ctx =   [n_tasks, k_shot, 1]
    # x_test = [n_tasks, n_datapoints, 1]
    # return y_pred = [n_tasks, n_samples, n_datapoints, 1]
    def predict(self, x_ctx, y_ctx, x_test):
        y_pred=torch.zeros((x_test.shape[0], self.config['num_models'], x_test.shape[1], 1))
        for i in range(0, self.config['num_test_tasks'], self.config['minibatch']):
            x_c=x_ctx[i:i+self.config['minibatch']]
            y_c=y_ctx[i:i+self.config['minibatch']]
            x_t=x_test[i:i+self.config['minibatch']]
            y_pred[i:i+self.config['minibatch']] = self.predict_minibatch(x_c, y_c, x_t)
        return y_pred

    # x_ctx, y_ctx =   [minibatch, k_shot, 1]
    # x_test, y_test = [minibatch, n_datapoints, 1]
    # return y_pred = [minibatch, n_samples,  n_datapoints, 1]
    def predict_minibatch(self, x_ctx, y_ctx, x_test):
        # there is slef.algo.adapt() method but it takes a MetaLearningTask as input. but predict only works with a whole minibatch as input.
        # if I input only one task [n_datapoints, 1] into predict() it outputs [minibatch, n_datapoints, 1] anyways
        y_pred = torch.zeros((self.config['minibatch'], self.config['num_models'], self.config['num_points_per_test_task'], 1))
        for i in range(self.config['num_models']):
            self.algo.adapt(x_ctx, y_ctx)
            y_p, _ = self.algo.predict(x_test)
            y_pred[:, i, :, :] = torch.tensor(y_p)
        return y_pred

    # y_pred = [n_tasks, n_samples, n_datapoints]
    # y_test = [n_tasks, 1, n_datapoints]
    def calculate_losses(self, y_pred, y_test):
        nlml=calculate_neg_log_marginal_likelihood(
            y_pred, y_test, torch.tensor(self.config['noise_stddev']))
        mse = torch.nn.MSELoss()
        y_test = torch.broadcast_to(y_test, y_pred.shape)
        mse_loss = mse(y_pred, y_test).item()
        if(self.config['wandb']):
            wandb.log({
                'evaluation/run_id': 1,
                'evaluation/nlml': nlml,
                'evaluation/mse': mse_loss
            })
        print(f"Test-NLML: {nlml}")
        print(f"Test-MSE: {mse_loss}")
