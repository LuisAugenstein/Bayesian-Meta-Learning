import wandb

def init(config: dict) -> None:
    wandb.init(project="fsml_" + config['algorithm'],
                       entity="seminar-meta-learning",
                       config=config)
    wandb.define_metric(name="meta_train/epoch")
    wandb.define_metric(name="meta_train/*",
                        step_metric="meta_train/epoch")

    wandb.define_metric(name="adapt/epoch")
    wandb.define_metric(name="adapt/*", step_metric="adapt/epoch")

    wandb.define_metric(name="results/sample")
    wandb.define_metric(name="results/*", step_metric="results/sample")
