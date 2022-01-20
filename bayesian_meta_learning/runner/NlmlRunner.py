from bayesian_meta_learning.runner.BaseRunner import BaseRunner
import numpy as np
from bayesian_meta_learning.benchmark.benchmark_dataloader import create_benchmark_dataloaders
from bayesian_meta_learning.nlml_tester import test_neg_log_marginal_likelihood

class NlmlRunner(BaseRunner):
    def __init__(self, config) -> None:
        super().__init__(config=config)

    def run(self) -> None:
        num_runs = 2
        nlml = [float] * num_runs
        for run_id in range(num_runs):
            train_dataloader, _, test_dataloader = create_benchmark_dataloaders(self.config)
            self.algo.train(train_dataloader, None)
            nlml[run_id] = test_neg_log_marginal_likelihood(self.algo, test_dataloader, self.config)
        mean = np.round(np.mean(nlml),4)
        std = np.round(np.std(nlml), 4)
        print(f"NLML: {mean} +- {std}")

        pass