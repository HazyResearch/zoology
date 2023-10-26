from pathlib import Path

import wandb
from torch.nn import Module

from zoology.config import LoggerConfig, TrainConfig

class WandbLogger:
    def __init__(self, config: TrainConfig):
        self.run = wandb.init(
            name=config.run_id,
            project=config.logger.project_name, 
        )
        wandb.run.log_code(
            root=str(Path(__file__).parent.parent),
            include_fn=lambda path, root: path.endswith(".py")
        )

    def log_config(self, config: TrainConfig):
        self.run.config.update(config.model_dump(), allow_val_change=True)

    def log_model(self, model: Module):
        wandb.watch(model)

    def log(self, metrics: dict):
        wandb.log(metrics)


