from distutils.command.config import config
import torch
import numpy as np
import higher
import typing
from tqdm import tqdm
import wandb

from few_shot_meta_learning.MLBaseClass import MLBaseClass
from few_shot_meta_learning.HyperNetClasses import EnsembleNet
from few_shot_meta_learning.Maml import Maml


class BmamlChaser(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

        self.hyper_net_class = EnsembleNet

    def load_model(self, resume_epoch: int, eps_dataloader: torch.utils.data.DataLoader, **kwargs) -> dict:
        maml_temp = Maml(config=self.config)
        return maml_temp.load_model(resume_epoch=resume_epoch, eps_dataloader=eps_dataloader, **kwargs)

    def adaptation(self, x: torch.Tensor, y: torch.Tensor, model: dict, is_leader: bool = False) -> typing.List[higher.patch._MonkeyPatchBase]:
        """"""
        f_hyper_net = higher.patch.monkeypatch(
            module=model["hyper_net"],
            copy_initial_weights=False,
            track_higher_grads=self.config["train_flag"]
        )

        q_params = torch.stack(
            tensors=[p for p in model["hyper_net"].parameters()])

        if is_leader:
            steps = self.config["advance_leader"]
        else:
            steps = self.config["num_inner_updates"]
        for i in range(steps):
            distance_NLL = torch.empty(size=(
                self.config["num_models"], model["hyper_net"].num_base_params), device=self.config["device"])

            loss_monitor = 0
            for particle_id in range(self.config["num_models"]):
                base_net_params = f_hyper_net.forward(i=particle_id)

                logits = model["f_base_net"].forward(x, params=base_net_params)

                loss_temp = self.config['loss_function'](
                    input=logits, target=y)

                loss_monitor += loss_temp.item() / self.config['num_models']

                if self.config["first_order"]:
                    grads = torch.autograd.grad(
                        outputs=loss_temp,
                        inputs=f_hyper_net.fast_params[particle_id],
                        retain_graph=True
                    )
                else:
                    grads = torch.autograd.grad(
                        outputs=loss_temp,
                        inputs=f_hyper_net.fast_params[particle_id],
                        create_graph=True
                    )

                distance_NLL[particle_id, :] = torch.nn.utils.parameters_to_vector(
                    parameters=grads)

            # log adaptation
            if self.config['num_inner_updates'] > 500 and ((i+1) % 500 == 0 or i == 0):
                if i == 0:
                    print(' ')
                print('Epoch {:<5} {:<10}'.format(
                    i+1, np.round(loss_monitor, 4)))

            kernel_matrix, grad_kernel, _ = self.get_kernel(params=q_params)

            q_params = q_params - \
                self.config["inner_lr"] * \
                (torch.matmul(kernel_matrix, distance_NLL) - grad_kernel)

            # update hyper-net
            f_hyper_net.update_params(
                params=[q_params[i, :] for i in range(self.config["num_models"])])

        return f_hyper_net

    def prediction(self, x: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> typing.List[torch.Tensor]:
        """"""
        logits = [None] * self.config["num_models"]

        for particle_id in range(self.config["num_models"]):
            base_net_params = adapted_hyper_net.forward(i=particle_id)

            logits[particle_id] = model["f_base_net"].forward(
                x, params=base_net_params)

        return logits

    def validation_loss(self, x: torch.Tensor, y: torch.Tensor, adapted_hyper_net: higher.patch._MonkeyPatchBase, model: dict) -> torch.Tensor:
        """
        Implements the chaser loss function
        """

        logits = self.prediction(
            x=x, adapted_hyper_net=adapted_hyper_net, model=model)

        print(len(logits))
        loss = 0

        for logits_ in logits:
            loss_temp = self.config['loss_function'](input=logits_, target=y)
            loss = loss + loss_temp

        loss = loss / len(logits)

        return loss

    def evaluation(self, x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, model: dict) -> typing.Tuple[float, float]:
        """
        """
        adapted_hyper_net = self.adaptation(x=x_t, y=y_t, model=model)

        logits = self.prediction(
            x=x_v, adapted_hyper_net=adapted_hyper_net, model=model)

        # classification loss
        loss = 0
        for logits_ in logits:
            loss = loss + \
                self.config['loss_function'](input=logits_, target=y_v)

        loss = loss / len(logits)

        y_pred = 0
        for logits_ in logits:
            y_pred = y_pred + torch.softmax(input=logits_, dim=1)

        y_pred = y_pred / len(logits)

        accuracy = (y_pred.argmax(dim=1) == y_v).float().mean().item()

        return loss.item(), accuracy * 100

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: typing.Optional[torch.utils.data.DataLoader]) -> None:
        """Train meta-learning model

        Args:
            eps_dataloader: the generator that generate episodes/tasks
        """
        print("Training is started.")
        print(f"Models are stored at {self.config['logdir']}.\n")

        print("{:<7} {:<10} {:<10}".format(
            'Epoch', 'NLL_train', 'NLL_validation'))

        # initialize/load model. Please see the load_model method implemented in each specific class for further information about the model
        model = self.load_model(
            resume_epoch=self.config['resume_epoch'], hyper_net_class=self.hyper_net_class, eps_dataloader=train_dataloader)
        model["optimizer"].zero_grad()

        # store initial model
        self.saveModel(model, 0.1)

        # initialize a tensorboard summary writer for logging
        # tb_writer = SummaryWriter(
        #     log_dir=self.config['logdir'],
        #     purge_step=self.config['resume_epoch'] * self.config['num_episodes_per_epoch'] // self.config['minibatch_print'] if self.config['resume_epoch'] > 0 else None
        # )

        try:
            for epoch_id in range(self.config['resume_epoch'], self.config['resume_epoch'] + self.config['num_epochs'], 1):
                loss_monitor = 0.
                progress = tqdm(enumerate(train_dataloader))

                leaders = []
                chasers = []
                for eps_count, eps_data in progress:
                    #print(f"EPOCH: {epoch_id}, EPS Count: {eps_count}")
                    #print(f"eps_data: {eps_data[1].shape}")
                    if (eps_count >= self.config['num_episodes_per_epoch']):
                        break

                    # split data into train and validation
                    split_data = self.config['train_val_split_function'](
                        eps_data=eps_data, k_shot=self.config['k_shot'])

                    # move data to GPU (if there is a GPU)
                    x_t = split_data['x_t'].to(self.config['device'])
                    y_t = split_data['y_t'].to(self.config['device'])
                    x_v = split_data['x_v'].to(self.config['device'])
                    y_v = split_data['y_v'].to(self.config['device'])

                    # -------------------------
                    # adaptation
                    # -------------------------
                    # Train the chaser on the training split only
                    chaser = self.adaptation(
                        x=x_t, y=y_t, model=model)
                    chasers.append(chaser)

                    # Train the leader on the entrie dataset
                    leader = self.adaptation(
                        x=eps_data[0].T, y=eps_data[1].T, model=model, is_leader=True)
                    leaders.append(leader)

                    # update meta-parameters
                    if ((eps_count + 1) % self.config['minibatch'] == 0):
                        print("Meta update")

                        loss = self.chaser_loss(chasers, leaders)
                        loss.backward()

                        model["optimizer"].step()
                        model["optimizer"].zero_grad()

                        chasers = []
                        leaders = []
                        
                        # monitoring
                        if (eps_count + 1) % self.config['minibatch_print'] == 0 or \
                           (epoch_id == 0 and (eps_count+1) / self.config['minibatch'] == 0):
                            loss_monitor = loss_monitor * \
                                self.config["minibatch"] / \
                                self.config["minibatch_print"]

                            # calculate step for Tensorboard Summary Writer
                            global_step = (
                                epoch_id * self.config['num_episodes_per_epoch'] + eps_count + 1) // self.config['minibatch_print']

                            # tb_writer.add_scalar(tag="Train_Loss", scalar_value=loss_monitor, global_step=global_step)
                            if self.config['wandb']:
                                wandb.log({
                                    'meta_train/epoch': global_step,
                                    'meta_train/train_loss': loss_monitor
                                })
                            loss_train = np.round(loss_monitor, 4)

                            progress.set_description(
                                f"Episode loss {loss_train}")

                            # reset monitoring variables
                            loss_monitor = 0.

                            # -------------------------
                            # Validation
                            # -------------------------
                            loss_val = '-'
                            if val_dataloader is not None and val_dataloader.dataset.n_tasks != 0:
                                # turn on EVAL mode to disable dropout
                                model["f_base_net"].eval()

                                loss_temp, accuracy_temp = self.evaluate(
                                    num_eps=self.config['num_episodes'],
                                    eps_dataloader=val_dataloader,
                                    model=model
                                )

                                loss_val = np.mean(loss_temp)
                                if self.config['wandb']:
                                    wandb.log({
                                        'meta_train/epoch': global_step,
                                        'meta_train/val_loss': loss_val
                                    })
                                loss_val = np.round(loss_val, 4)

                                # tb_writer.add_scalar(tag="Val_NLL", scalar_value=np.mean(loss_temp), global_step=global_step)
                                # tb_writer.add_scalar(tag="Val_Accuracy", scalar_value=np.mean(accuracy_temp), global_step=global_step)

                                model["f_base_net"].train()
                                del loss_temp
                                del accuracy_temp

                # print on console
                print("Episode {:<3}: Validation Loss = {:<10}".format(
                    epoch_id+1, loss_monitor))

                # save model
                self.saveModel(model, epoch_id+1)
            print('Training is completed.\n')
        finally:
            pass
            # print('\nClose tensorboard summary writer')
            # tb_writer.close()

        return None

    def get_kernel(self, params: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the RBF kernel for the input

        Args:
            params: a tensor of shape (N, M)

        Returns: kernel_matrix = tensor of shape (N, N)
        """
        pairwise_d_matrix = self.get_pairwise_distance_matrix(x=params)

        # tf.reduce_mean(euclidean_dists) ** 2
        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)
        h = median_dist / np.log(self.config["num_models"])

        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        grad_kernel = -torch.matmul(kernel_matrix, params)
        grad_kernel += params * kernel_sum
        grad_kernel /= h

        return kernel_matrix, grad_kernel, h

    @staticmethod
    def get_pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
        """Calculate the pairwise distance between each row of tensor x

        Args:
            x: input tensor

        Return: matrix of point-wise distances
        """
        n, m = x.shape

        # initialize matrix of pairwise distances as a N x N matrix
        pairwise_d_matrix = torch.zeros(size=(n, n), device=x.device)

        # num_particles = particle_tensor.shape[0]
        euclidean_dists = torch.nn.functional.pdist(
            input=x, p=2)  # shape of (N)

        # assign upper-triangle part
        triu_indices = torch.triu_indices(row=n, col=n, offset=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        # assign lower-triangle part
        pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        return pairwise_d_matrix

    def chaser_loss(self, chasers: typing.List[higher.patch.monkeypatch], leaders: typing.List[higher.patch.monkeypatch]) -> torch.tensor:
        loss = 0
        # iterate over tasks
        for chaser_set, leader_set in zip(chasers, leaders):
            # iterate over particles
            for i in range(self.config['num_models']):
                chaser_particle = chaser_set.params[i]
                leader_particle = leader_set.params[i]
                leader_particle.detach()

                loss += torch.norm(chaser_particle - leader_particle)

        print(f"LOSS: {loss}")
        return -loss
