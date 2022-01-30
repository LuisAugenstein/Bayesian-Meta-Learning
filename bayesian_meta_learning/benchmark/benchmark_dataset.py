import torch
import typing
import numpy as np
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark

"""
    This Dataset is a wrapper for a benchmark (all tasks)
"""


class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, benchmark: MetaLearningBenchmark, normalize_enabled:bool) -> None:
        super().__init__()
        self.n_tasks = benchmark.n_task
        self.normalize_eanbled = normalize_enabled
        # calculate normalizers
        x = torch.zeros((benchmark.n_task, benchmark.n_datapoints_per_task))
        y = torch.zeros((benchmark.n_task, benchmark.n_datapoints_per_task))
        for i in range(benchmark.n_task):
            task = benchmark.get_task_by_index(i)
            x[i] = torch.tensor(task.x, dtype=torch.float32).squeeze()
            y[i] = torch.tensor(task.y, dtype=torch.float32).squeeze()
        self.normalizers = {
            "mean_x": x.mean(axis=(0, 1)),
            "mean_y": y.mean(axis=(0, 1)),
            "std_x": x.std(axis=(0, 1)),
            "std_y": y.std(axis=(0, 1)),
        }
        # normalize tasks
        self.x, self.y = self.normalize(x,y)
            

    def __len__(self) -> int:
        return 100000

    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        x = self.x[index % self.n_tasks]
        y = self.y[index % self.n_tasks]
        return [x, y]

    def normalize(self, x: typing.Optional[torch.Tensor] = None, y: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize_eanbled:
            return x,y
        normalized_x = x
        if x is not None and not (self.normalizers["std_x"] == 0.0).any():
            normalized_x = (x - self.normalizers['mean_x']) / self.normalizers['std_x']
        normalized_y = y
        if y is not None and not (self.normalizers["std_y"] == 0.0).any():
            normalized_y = (y - self.normalizers['mean_y']) / self.normalizers['std_y']
        return normalized_x, normalized_y

    def denormalize(self, x: typing.Optional[torch.Tensor] = None, y: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize_eanbled:
            return x,y
        denormalized_x = x
        if x is not None and not (self.normalizers["std_x"] == 0.0).any():
            denormalized_x = x * self.normalizers['std_x'] + self.normalizers['mean_x']
        denormalized_y = y
        if y is not None and not (self.normalizers["std_y"] == 0.0).any():
            denormalized_y = y * self.normalizers['std_y'] + self.normalizers['mean_y']
        return denormalized_x, denormalized_y
        
