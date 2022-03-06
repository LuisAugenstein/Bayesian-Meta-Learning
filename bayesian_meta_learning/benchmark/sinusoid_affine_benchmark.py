import numpy as np
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark, MetaLearningTask
from metalearning_benchmarks import Sinusoid, Affine1D

class CustomAffine1D(Affine1D):
    m_bounds = np.array([-2, 2])
    b_bounds = np.array([-2.5, 2.5])
    param_bounds = np.array([m_bounds, b_bounds])
    x_bounds = np.array([[-5.0, 5.0]])

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        super().__init__(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )

class SinusoidAffineBenchmark(MetaLearningBenchmark):
    d_param = 2
    d_x = 1
    d_y = 1

    # x_bounds depend on the task (Sinusoid or Affine) and are defined there
    x_bounds = np.array([0, 0])     

    def __init__(self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise):
        super().__init__(n_task, n_datapoints_per_task,
                         output_noise, seed_task, seed_x, seed_noise)
        self.n_sinusoids = n_task//2
        self.n_affines = n_task - self.n_sinusoids
        self.sinusoid = Sinusoid(
            self.n_sinusoids, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise)
        self.affine1D = CustomAffine1D(self.n_affines, n_datapoints_per_task,
                                 output_noise, seed_task, seed_x, seed_noise)
        rng = np.random.RandomState(seed_task)
        self.permute = rng.permutation(n_task)

    """
        returns either a Sinusoid Task or an Affine1D Task depending on the task_index
    """
    def _get_task_by_index_without_noise(self, task_index):
        permuted_index = self.permute[task_index]
        bm, i = (self.sinusoid, permuted_index) if permuted_index < self.n_sinusoids else (
            self.affine1D, permuted_index - self.n_sinusoids)
        return MetaLearningTask(x=bm.x[i], y=bm.y[i], param=bm.params[i])
