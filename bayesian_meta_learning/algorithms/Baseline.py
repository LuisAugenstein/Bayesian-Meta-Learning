import torch
import wandb
import typing
import numpy as np
from few_shot_meta_learning.Maml import Maml
from tqdm import tqdm

class Baseline(Maml):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)


    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: typing.Optional[torch.utils.data.DataLoader]) -> None:
        #assert len(train_dataloader.dataset) == self.config['minibatch']
        print("Training is started.")
        print(f"Models are stored at {self.config['logdir']}.\n")

        print("{:<10}: train loss for the current minibatch".format('NLL_train'))
        print("{:<10}: val loss for all tasks in the validation set\n".format('NLL_val'))

        print("{:<6} {:<10} {:<10} {:<10}".format(
            'Epoch', 'Minibatch', 'NLL_train', 'NLL_val'))

        # set num_inner_updates to 0 for the training 
        num_inner_updates = self.config['num_inner_updates']
        self.config['num_inner_updates'] = 0

        # initialize/load model.
        model = self.load_model(
            resume_epoch=self.config['resume_epoch'], hyper_net_class=self.hyper_net_class, eps_dataloader=train_dataloader)
        model["optimizer"].zero_grad()

        # store initial model
        self.saveModel(model, 0.1)

        try:
            for epoch_id in range(self.config['resume_epoch'], self.config['resume_epoch'] + self.config['num_epochs'], 1):
                loss_monitor = 0.
                progress = tqdm(enumerate(train_dataloader))
                for eps_count, eps_data in progress:
                    if (eps_count >= self.config['num_episodes_per_epoch']):
                        break

                    eps_data_batch = [
                        eps_data[i].T for i in range(len(eps_data))]
                    x = eps_data_batch[0].to(self.config['device'])
                    y = eps_data_batch[1].to(self.config['device'])

                    # -------------------------
                    # adaptation on training subset
                    # -------------------------
                    non_adapted_hyper_net = self.adaptation(x, y, model)

                    # -------------------------
                    # loss on validation subset
                    # -------------------------
                    loss_v = self.validation_loss(
                        x=x, y=y, adapted_hyper_net=non_adapted_hyper_net, model=model)
                    loss_v = loss_v / self.config["minibatch"]

                    if torch.isnan(input=loss_v):
                        raise ValueError("Loss is NaN.")

                    # calculate gradients w.r.t. hyper_net's parameters
                    loss_v.backward()

                    loss_monitor += loss_v.item()
                    # update meta-parameters
                    if ((eps_count + 1) % self.config['minibatch'] == 0):
                        model["optimizer"].step()
                        model["optimizer"].zero_grad()

                        # monitoring
                        if (eps_count + 1) % self.config['minibatch_print'] == 0:
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
                            # reset monitoring variables
                            loss_monitor = 0.

                            # -------------------------
                            # Validation
                            # -------------------------
                            loss_val = 0
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
                                        'meta_train/val_loss': loss_val
                                    })
                                loss_val = np.round(loss_val, 4)
                                # tb_writer.add_scalar(tag="Val_NLL", scalar_value=np.mean(loss_temp), global_step=global_step)
                                # tb_writer.add_scalar(tag="Val_Accuracy", scalar_value=np.mean(accuracy_temp), global_step=global_step)
                                model["f_base_net"].train()
                                del loss_temp
                                del accuracy_temp
                            # plot train and val loss with tqdm
                            minibatch_number = (
                                eps_count + 1) // self.config["minibatch_print"]
                            loss_string = "{:<6} {:<10} {:<10} {:<10}".format(
                                epoch_id+1, minibatch_number, loss_train, loss_val)
                            progress.set_description(loss_string)

                if (epoch_id + 1) % self.config['epochs_to_save'] == 0:
                    # save model
                    self.saveModel(model, epoch_id+1)
            print('Training is completed.\n')
        finally:
            self.config['num_inner_updates'] = num_inner_updates
            # print('\nClose tensorboard summary writer')
            # tb_writer.close()

        return None
