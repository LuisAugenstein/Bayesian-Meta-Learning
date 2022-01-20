from typing import Tuple
import wandb
import torch
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
import random
import matplotlib.pyplot as plt



def plot_tasks_initially(caption, algo, task_dataloader: DataLoader, config):
    if task_dataloader.dataset.n_tasks == 0:
        return
    model = model = algo.load_model(
        resume_epoch=0.1, hyper_net_class=algo.hyper_net_class, eps_dataloader=task_dataloader)
    plotting_data = [None] * task_dataloader.dataset.n_tasks
    for task_index in range(task_dataloader.dataset.n_tasks):
        task_data = task_dataloader.dataset[task_index]
        x_train, y_train, x_test, y_test = _split_data(task_data, config)
        # set seeds for platipus so that the samples phi are drawn equally for each dataset
        if config['algorithm'] == 'platipus':
            torch.manual_seed(123)
            torch.cuda.manual_seed(123)
        # plot prediction of the initial model
        phi = algo.adaptation(x_train[:, None], y_train[:, None], model)
        y_pred = algo.prediction(x_test, phi, model)
        if config['algorithm'] == 'platipus' or config['algorithm'] == 'bmaml':
            # platipus/bmaml return no tensor but a list of S tensors
            y_pred = torch.stack(y_pred)
        plotting_data[task_index] = {
            'x_train': x_train.squeeze().cpu().detach().numpy(),
            'y_train': y_train.squeeze().cpu().detach().numpy(),
            'x_test': x_test.squeeze().cpu().detach().numpy(),
            'y_test': y_test.squeeze().cpu().detach().numpy(),
            'y_pred': y_pred.squeeze().cpu().detach().numpy()
        }
    _generate_task_initially_plots(caption, plotting_data, config)


def plot_task_results(caption, epoch, algo, task_dataloader, config):
    if task_dataloader.dataset.n_tasks == 0:
        return
    model = model = algo.load_model(
        resume_epoch=epoch, hyper_net_class=algo.hyper_net_class, eps_dataloader=task_dataloader)
    num_visualization_tasks = np.min(
        [config['num_visualization_tasks'], task_dataloader.dataset.n_tasks])
    plotting_data = [None] * num_visualization_tasks
    for task_index, task_data in enumerate(task_dataloader):
        if task_index >= num_visualization_tasks:
            break
        # samples and true target function of the task
        x_train, y_train, x_test, y_test = _split_data(task_data, config)
        # approximate posterior predictive distribution
        y_pred, heat_map, y_resolution = _predict_test_data(
            algo, model, x_train, y_train, x_test, y_test, config)
        plotting_data[task_index] = {
            'x_train': x_train.squeeze().cpu().detach().numpy(),
            'y_train': y_train.squeeze().cpu().detach().numpy(),
            'x_test': x_test.squeeze().cpu().detach().numpy(),
            'y_test': y_test.squeeze().cpu().detach().numpy(),
            'y_pred': y_pred.squeeze().detach().numpy(),
            'heat_map': heat_map.cpu().detach().numpy(),
            'y_resolution': y_resolution.detach().numpy(),
        }
    # plot the plotting data
    _generate_plots(caption, epoch, plotting_data, config)


def _split_data(task_data, config):
    task_data_T = [task_data[i].T for i in range(len(task_data))]
    x_test, sort_indices = torch.sort(task_data_T[0].squeeze())
    x_test = x_test[:, None]
    y_test = task_data_T[1][sort_indices]
    # use random seed to draw the k-shot samples equal for all evaluations
    random.seed(config['seed'])
    # generate training samples and move them to GPU (if there is a GPU)
    split_data = config['train_val_split_function'](
        eps_data=task_data, k_shot=config['k_shot'])
    x_train = split_data['x_t'].to(config['device'])
    y_train = split_data['y_t'].to(config['device'])
    return x_train, y_train, x_test, y_test


def _predict_test_data(algo, model, x_train, y_train, x_test, y_test, config):
    S = 1 if config['algorithm'] == 'maml' else config['num_models']
    N = x_test.shape[0]  # equals points_per_minibatch for the considered task
    R = config['y_plotting_resolution']
    noise_var = config['noise_stddev']**2

    # platipus uses random samples theta. we want them to be always the same
    if config['algorithm'] == 'platipus':
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
    adapted_hyper_net = algo.adaptation(x_train, y_train, model)
    # predict x_test
    y_pred = algo.prediction(x_test, adapted_hyper_net, model)
    if config['algorithm'] == 'platipus' or config['algorithm'] == 'bmaml':
        # platipus/bmaml return no tensor but a list of S tensors
        y_pred = torch.stack(y_pred)
    y_pred = torch.broadcast_to(y_pred, (S, N, 1))

    # discretize the relevant space of y-values
    y_combined = torch.concat([y_test[None, :], y_pred])
    start, end = (torch.min(y_combined).data, torch.max(y_combined).data)
    y_resolution = torch.linspace(start, end, R)
    y_broadcasted = torch.broadcast_to(y_resolution, (1, N, R))

    # generate heat_map with density values at the discretized points
    heat_maps = torch.exp(-(y_broadcasted-y_pred)**2/(
        2*noise_var)) / np.sqrt(2*torch.pi*noise_var)
    heat_map = torch.mean(heat_maps, axis=0)
    heat_map = heat_map[1:, 1:].T
    return y_pred, heat_map, y_resolution


# ==============================================
# =================Plotting=====================
# ==============================================

def _index_to_row_col(i: int) -> Tuple[int, int]:
    n_rows = np.floor(np.sqrt(i))
    n_cols = np.ceil(i / n_rows)
    return int(n_rows), int(n_cols)


def _generate_task_initially_plots(caption, plotting_data, config):
    n_rows, n_cols = _index_to_row_col(len(plotting_data))
    fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)
    for i, data in enumerate(plotting_data):
        row = i // n_cols
        col = i % n_cols
        _plot_samples(data, axs[row, col])
    fig.suptitle(
        f"Benchmark={config['benchmark']}, num_points_per_train_task={config['num_points_per_train_task']}")
    fig.set_figwidth(12)
    plt.tight_layout()
    # save the plot
    _save_plot(caption=caption, index="", config=config)


def _generate_plots(caption, epoch, plotting_data, config):
    fig, axs = plt.subplots(2, len(plotting_data), squeeze=False)
    # plot the data
    for i, data in enumerate(plotting_data):
        _plot_distribution(data, axs[0, i], fig)
        _plot_samples(data, axs[1, i])
    # add cosmetics
    num_models = 1 if config['algorithm'] == 'maml' else config['num_models']
    fig.suptitle(
        f"{config['algorithm'].upper()} - {config['benchmark']} - {epoch} epochs \n" +
        f"k_shot={config['k_shot']}, noise_sttdev={config['noise_stddev']}, num_models={num_models}, \n" +
        f"num_points_per_test_task={config['num_points_per_test_task']} ")

    fig.set_figwidth(12)
    plt.tight_layout()
    # save the plot
    _save_plot(caption, index=f"Epoch_{epoch}", config=config)


def _save_plot(caption, index, config):
    if config['wandb']:
        wandb.log({caption: wandb.Image(plt, index)})
    else:
        filename = f"{caption}" if index == "" else f"{caption}-{index}"
        save_path = os.path.join(config['logdir_plots'], filename)
        plt.savefig(save_path)
        print(f"stored plot: {filename}")


def _plot_distribution(data, ax, fig):
    _base_plot(data, ax)
    # plot posterior predictive distribution
    max_heat = np.max(data['heat_map'])
    min_heat = np.min(data['heat_map'])
    c = ax.pcolormesh(data['x_test'], data['y_resolution'],
                      data['heat_map'], vmin=min_heat, vmax=max_heat)
    fig.colorbar(c, ax=ax)


def _plot_samples(data, ax):
    _base_plot(data, ax)
    # plot samples
    if data['y_pred'].shape == data['x_test'].shape:
        ax.plot(data['x_test'], data['y_pred'], linestyle='--')
        return
    for i in range(data['y_pred'].shape[0]):
        ax.plot(data['x_test'], data['y_pred'][i, :], linestyle='--')


def _base_plot(data, ax):
    # plot ground truth
    ax.plot(data['x_test'], data['y_test'], color='black',
            linewidth=1, linestyle='-')
    # plot samples
    ax.scatter(x=data['x_train'], y=data['y_train'],
               s=40, marker='^', color='C3', zorder=2, alpha=0.75)
    # additional information
    ax.set_xlabel('x')
    ax.set_ylabel('y')
