
import torch
import numpy as np
import random
from abc import ABC, abstractmethod
from bayesian_meta_learning.benchmark.benchmark_dataloader import create_benchmark_dataloaders
from bayesian_meta_learning.algorithms import Baseline
from few_shot_meta_learning.Maml import Maml
from few_shot_meta_learning.Platipus import Platipus
from few_shot_meta_learning.Bmaml import Bmaml

class BaseRunner(ABC):

    def __init__(self, config: dict) -> None:
        self.config = config
        # set seeds
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        # create dataloaders
        self.train_dataloader, self.val_dataloader, self.test_dataloader = create_benchmark_dataloaders(
            config)
        # initialize algorithm to use
        algorithms = {
            'maml': Maml,
            'bmaml': Bmaml,
            'platipus': Platipus,
            'baseline': Baseline
        }
        self.algo = algorithms[config['algorithm']](config)
        return

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()