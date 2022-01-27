from bayesian_meta_learning.runner.BaseRunner import BaseRunner
import numpy as np
from bayesian_meta_learning.benchmark.benchmark_dataloader import create_benchmark_dataloaders
from bayesian_meta_learning.nlml_tester import test_neg_log_marginal_likelihood
import wandb


class NlmlRunner(BaseRunner):
    def __init__(self, config) -> None:
        super().__init__(config=config)

    def run(self) -> None:
        train_dataloader, _, test_dataloader = create_benchmark_dataloaders(
            self.config)
        self.algo.train(train_dataloader, None)

        if self.config['wandb']:
            wandb.define_metric(name="nlml_test/run_id")
            wandb.define_metric(name="nlml_test/loss", step_metric="nlml_test/run_id")

        num_runs = 100 if self.config['algorithm'] == 'platipus' else 1
        nlml = [float] * num_runs
        for run_id in range(num_runs):
            nlml[run_id] = test_neg_log_marginal_likelihood(
                self.algo, test_dataloader, self.config)
            if(self.config['wandb']):
                wandb.log({
                    'nlml_test/run_id': run_id,
                    'nlml_test/loss': nlml[run_id]
                })
            print(f"{run_id}: {np.round(nlml[run_id], 4)}")
        mean = np.round(np.mean(nlml), 4)
        std = np.round(np.std(nlml), 4)
        print(f"\nNLML: {mean} +- {std}")

        pass
