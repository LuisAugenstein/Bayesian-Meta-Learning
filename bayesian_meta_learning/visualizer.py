import os
import wandb
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple
from torch.utils.data.dataloader import DataLoader

def plot_tasks_initially(caption, algo, task_dataloader: DataLoader, config):
    if task_dataloader.dataset.n_tasks == 0:
        return
    model = model = algo.load_model(
        resume_epoch=0.1, hyper_net_class=algo.hyper_net_class, eps_dataloader=task_dataloader)
    num_visualization_tasks = np.min([task_dataloader.dataset.n_tasks, 20])
    y_pred, y_test, x_test, y_train, x_train = _predict_all_tasks(num_visualization_tasks,
                                                                  algo, model, task_dataloader, config)
    # denormalize data
    _, y_pred = task_dataloader.dataset.denormalize(y=y_pred)
    x_test, y_test = task_dataloader.dataset.denormalize(x=x_test, y=y_test)
    x_train, y_train = task_dataloader.dataset.denormalize(
        x=x_train, y=y_train)
    # generate plotting data
    plotting_data = [None] * num_visualization_tasks
    for task_index in range(num_visualization_tasks):
        plotting_data[task_index] = {
            'x_train': x_train[task_index].squeeze().cpu().detach().numpy(),
            'y_train': y_train[task_index].squeeze().cpu().detach().numpy(),
            'x_test': x_test[task_index].squeeze().cpu().detach().numpy(),
            'y_test': y_test[task_index].squeeze().cpu().detach().numpy(),
            'y_pred': y_pred[task_index].squeeze().cpu().detach().numpy()
        }
    _generate_task_initially_plots(caption, plotting_data, config)


def plot_task_results(caption, epoch, algo, task_dataloader, config):
    if task_dataloader.dataset.n_tasks == 0:
        return
    model = model = algo.load_model(
        resume_epoch=epoch, hyper_net_class=algo.hyper_net_class, eps_dataloader=task_dataloader)
    num_visualization_tasks = np.min(
        [config['num_visualization_tasks'], task_dataloader.dataset.n_tasks])
    num_tasks_to_plot = task_dataloader.dataset.n_tasks - \
        (task_dataloader.dataset.n_tasks %
         num_visualization_tasks) if caption == 'Testing' else num_visualization_tasks
    y_pred, y_test, x_test, y_train, x_train = _predict_all_tasks(num_tasks_to_plot,
                                                                  algo, model, task_dataloader, config)
    # denormalize data
    _, y_pred = task_dataloader.dataset.denormalize(y=y_pred)
    x_test, y_test = task_dataloader.dataset.denormalize(x=x_test, y=y_test)
    x_train, y_train = task_dataloader.dataset.denormalize(
        x=x_train, y=y_train)
    # create plotting data
    if(caption == 'Testing'):
        for i in range(0, num_tasks_to_plot, num_visualization_tasks):
            plotting_data = generate_plotting_data(
                y_pred[i:i+num_visualization_tasks],
                y_test[i:i+num_visualization_tasks],
                x_test[i:i+num_visualization_tasks],
                y_train[i:i+num_visualization_tasks],
                x_train[i:i+num_visualization_tasks], num_visualization_tasks, config)
            # plot the plotting data
            generate_plots(
                f"{caption}_{i // num_visualization_tasks}", epoch, plotting_data, config)
    else:
        plotting_data = generate_plotting_data(
            y_pred, y_test, x_test, y_train, x_train, num_visualization_tasks, config)
        # plot the plotting data
        generate_plots(caption, epoch, plotting_data, config)


def generate_plotting_data(y_pred, y_test, x_test, y_train, x_train, num_visualization_tasks, config):
    plotting_data = [None] * num_visualization_tasks
    for task_index in range(num_visualization_tasks):
        R = config['y_plotting_resolution']
        S = y_pred.shape[1]
        N = y_pred.shape[2]
        y_combined = torch.cat([y_test[task_index], y_pred[task_index]])
        start, end = (torch.min(y_combined).data, torch.max(y_combined).data)
        y_resolution = torch.linspace(start, end, R)
        y_broadcasted = torch.broadcast_to(y_resolution, (1, N, R))
        y_p = torch.broadcast_to(y_pred[task_index, :, :, None], (S, N, 1))
        # move to cpu
        y_p = y_p.cpu()
        y_broadcasted = y_broadcasted.cpu()
        y_resolution = y_resolution.cpu()

        # generate heat_map with density values at the discretized points
        noise_var = config['noise_stddev']**2
        heat_maps = torch.exp(-(y_broadcasted-y_p)**2/(
            2*noise_var)) / np.sqrt(2*np.pi*noise_var)
        heat_map = torch.mean(heat_maps, axis=0)
        heat_map = heat_map[1:, 1:].T
        plotting_data[task_index] = {
            'x_train': x_train[task_index].squeeze().cpu().detach().numpy(),
            'y_train': y_train[task_index].squeeze().cpu().detach().numpy(),
            'x_test': x_test[task_index].squeeze().cpu().detach().numpy(),
            'y_test': y_test[task_index].squeeze().cpu().detach().numpy(),
            'y_pred': y_pred[task_index].squeeze().cpu().detach().numpy(),
            'heat_map': heat_map.cpu().detach().numpy(),
            'y_resolution': y_resolution.cpu().detach().numpy(),
        }
    return plotting_data


def _predict_all_tasks(n_tasks: int, algo, model, task_dataloader, config: dict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    S = 1 if config['algorithm'] == 'maml' else config['num_models']
    T = task_dataloader.dataset.n_tasks
    N = task_dataloader.dataset[0][0].shape[0]
    y_pred = torch.zeros((T, S, N)).to(config['device'])
    y_test = torch.zeros((T, 1, N)).to(config['device'])
    x_test = torch.zeros((T, 1, N)).to(config['device'])
    x_train = torch.zeros((T, 1, config['k_shot'])).to(config['device'])
    y_train = torch.zeros((T, 1, config['k_shot'])).to(config['device'])
    for task_index in range(np.min([n_tasks, task_dataloader.dataset.n_tasks])):
        task_data = task_dataloader.dataset[task_index]
        # split the data
        x_test_t, sort_indices = torch.sort(task_data[0])
        y_test_t = task_data[1][sort_indices]

        # Move to gpu if available
        x_test_t = x_test_t.to(config['device'])
        y_test_t = y_test_t.to(config['device'])

        # use random seed to draw the k-shot samples equal for all evaluations
        random.seed(config['seed_offset_test'])
        # generate training samples and move them to GPU (if there is a GPU)
        split_data = config['train_val_split_function'](
            eps_data=task_data, k_shot=config['k_shot'])
        x_train_t = split_data['x_t'].to(config['device'])
        y_train_t = split_data['y_t'].to(config['device'])
        # set seeds for platipus so that the samples phi are drawn equally for each dataset
        if config['algorithm'] == 'platipus':
            torch.manual_seed(123)
            torch.cuda.manual_seed(123)
        # plot prediction of the initial model
        phi = algo.adaptation(x_train_t[:, None], y_train_t[:, None], model)
        y_pred_t = algo.prediction(x_test_t[:, None], phi, model)
        if config['algorithm'] == 'platipus' or config['algorithm'] == 'bmaml' or config['algorithm'] == 'bmaml_chaser':
            # platipus/bmaml return no tensor but a list of S tensors
            y_pred_t = torch.stack(y_pred_t)
        y_pred[task_index] = y_pred_t.squeeze()
        y_test[task_index] = y_test_t
        x_test[task_index] = x_test_t
        x_train[task_index] = x_train_t
        y_train[task_index] = y_train_t
    return y_pred, y_test, x_test, y_train, x_train


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


def generate_plots(caption, epoch, plotting_data, config):
    fig, axs = plt.subplots(1, len(plotting_data),
                            squeeze=False, figsize=(12, 3))
    # plot the data
    for i, data in enumerate(plotting_data):
        _plot_distribution(data, axs[0, i], fig)
        #_plot_samples(data, axs[1, i])
    # add cosmetics
    num_models = 1 if config['algorithm'] == 'maml' else config['num_models']

    # fig.set_figwidth(12)
    plt.tight_layout()
    # save the plot
    _save_plot(caption+"_nlml", index=f"Epoch_{epoch}", config=config)

    fig, axs = plt.subplots(1, len(plotting_data),
                            squeeze=False, figsize=(12, 3))
    # plot the data
    for i, data in enumerate(plotting_data):
        #_plot_distribution(data, axs[0, i], fig)
        _plot_samples(data, axs[0, i])
    # add cosmetics
    num_models = 1 if config['algorithm'] == 'maml' else config['num_models']

    # fig.set_figwidth(12)
    plt.tight_layout()
    # save the plot
    _save_plot(caption+"_plot", index=f"Epoch_{epoch}", config=config)

# caption: main name of the file
# index: Epoch_number


def _save_plot(caption: str, index: str, config: dict):
    filename = caption if index == "" else f"{caption}_{index}"
    if config['wandb']:
        wandb.log({caption: wandb.Image(plt, index)})
        print(f"stored to wandb: {filename}")

        save_path = os.path.join(config['logdir_plots'], filename + ".pdf")
        plt.savefig(save_path)
        print(f"stored: {filename}")
    else:
        save_path = os.path.join(config['logdir_plots'], filename)
        plt.savefig(save_path)
        print(f"stored: {filename}")


def _plot_distribution(data, ax, fig):
    _base_plot(data, ax, color='black')
    # plot posterior predictive distribution
    max_heat = np.max(data['heat_map'])
    min_heat = np.min(data['heat_map'])
    c = ax.pcolormesh(data['x_test'],
                      data['y_resolution'],
                      data['heat_map'],
                      vmin=min_heat,
                      vmax=max_heat,
                      cmap="BuPu"
                      )
    #fig.colorbar(c, ax=ax)


def _plot_samples(data, ax):
    _base_plot(data, ax)
    # plot samples
    if data['y_pred'].shape == data['x_test'].shape:
        ax.plot(data['x_test'], data['y_pred'], linestyle='--')
        return
    for i in range(data['y_pred'].shape[0]):
        ax.plot(data['x_test'], data['y_pred'][i, :], linestyle='--')


def _base_plot(data, ax, color='black'):
    # plot ground truth
    ax.plot(data['x_test'], data['y_test'], color=color,
            linewidth=1, linestyle='-')
    # additional information
    ax.set_xlabel('input [x]')
    ax.set_ylabel('output [y]')
    # plot samples
    ax.scatter(x=data['x_train'], y=data['y_train'],
               s=30, marker='^', color='C3', zorder=9, alpha=0.75)
