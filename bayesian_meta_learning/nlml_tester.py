
import torch
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from typing import Tuple
import numpy as np


def test_neg_log_marginal_likelihood(algo, test_dataloader: DataLoader, config: dict) -> float:
    print("Testing is started.")
    model = algo.load_model(
        resume_epoch=config["num_epochs"], hyper_net_class=algo.hyper_net_class, eps_dataloader=test_dataloader)
    nlml_per_task = torch.zeros((test_dataloader.dataset.n_tasks))
    y_pred, y_test = _predict_all_tasks(algo, model, test_dataloader, config)
    N = y_pred.shape[2]
    S = y_pred.shape[1]
    noise_var = config['noise_stddev']**2
    #constant = N/2*np.log(2*np.pi*noise_var) + np.log(S)
    constant = 0
    for task_index in range(test_dataloader.dataset.n_tasks):
        exponent = torch.norm(y_pred[task_index] - y_test[task_index],
                              dim=1)**2 / (2*noise_var)
        nlml_per_task[task_index] = constant - \
            torch.logsumexp(-exponent, dim=0)
    nlml = torch.mean(nlml_per_task).item()
    print(f"NLML: {nlml}\n")
    return nlml


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
            y_pred_t = torch.stack(y_pred)
        y_pred[task_index] = y_pred_t.squeeze()
        y_test[task_index] = y_test_t
    return y_pred, y_test
