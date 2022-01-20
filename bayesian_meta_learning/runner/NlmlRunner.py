from bayesian_meta_learning.runner.BaseRunner import BaseRunner


class MainRunner(BaseRunner):
    def __init__(self, config) -> None:
        super().__init__(config=config)

    def run(self) -> None:
        # run the algorithms several times and evaluate their NLML. 
        # average the nlml values over the different runs and compute sample mean and standard deviation
        # 
        # for platipus its enough to train once and then sample several times from the posterior 
        # Maml and BMaml need to be trained each run from zero 
        pass