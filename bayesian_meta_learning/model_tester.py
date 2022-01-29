
import torch
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from typing import Tuple
import numpy as np


def calculate_loss_metrics(algo, test_dataloader: DataLoader, config: dict) -> Tuple[float, float]:
    model = algo.load_model(
        resume_epoch=config["num_epochs"], hyper_net_class=algo.hyper_net_class, eps_dataloader=test_dataloader)
    y_pred, y_test = _predict_all_tasks(algo, model, test_dataloader, config)
    _, y_pred = test_dataloader.dataset.denormalize(y=y_pred)
    _, y_test = test_dataloader.dataset.denormalize(y=y_test)
    nlml = _calculate_neg_log_marginal_likelihood(
        y_pred, y_test, torch.tensor(config['noise_stddev']))
    mse = torch.nn.MSELoss()
    y_test = torch.broadcast_to(y_test, y_pred.shape)
    mse_loss = mse(y_pred, y_test).item()
    return nlml, mse_loss


def _calculate_neg_log_marginal_likelihood(y_pred: torch.Tensor, y_test: torch.Tensor, noise_stddev: torch.Tensor) -> float:
    T = y_pred.shape[0]  # num tasks
    S = y_pred.shape[1]  # num model samples
    N = y_pred.shape[2]  # num datapoints
    gaussian = torch.distributions.Normal(y_pred, noise_stddev)
    log_prob = torch.sum(gaussian.log_prob(y_test), dim=2)
    nlml_per_task = (np.log(S) - torch.logsumexp(log_prob, dim=1)) / N
    nlml = torch.sum(nlml_per_task) / T
    return nlml.item()


def _predict_all_tasks(algo, model, task_dataloader, config: dict) -> Tuple[Tensor, Tensor]:
    S = 1 if config['algorithm'] == 'maml' else config['num_models']
    T = task_dataloader.dataset.n_tasks
    N = config['num_points_per_test_task'] - config['k_shot']
    y_pred = torch.zeros((T, S, N))
    y_test = torch.zeros((T, 1, N))
    for task_index in range(task_dataloader.dataset.n_tasks):
        task_data = task_dataloader.dataset[task_index]
        # split the data
        split_data = config['train_val_split_function'](
            eps_data=task_data, k_shot=config['k_shot'])
        x_train_t = split_data['x_t'].to(config['device'])
        y_train_t = split_data['y_t'].to(config['device'])
        x_test_t = split_data['x_v'].to(config['device'])
        y_test_t = split_data['y_v'].to(config['device'])
        # adapt model and predict test set
        phi = algo.adaptation(x_train_t[:, None], y_train_t[:, None], model)
        y_pred_t = algo.prediction(x_test_t[:, None], phi, model)

        if config['algorithm'] == 'platipus' or config['algorithm'] == 'bmaml':
            # platipus/bmaml return no tensor but a list of S tensors
            y_pred_t = torch.stack(y_pred_t)
        y_pred[task_index] = y_pred_t.squeeze()
        y_test[task_index] = y_test_t
    return y_pred, y_test
