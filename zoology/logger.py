from pathlib import Path

import wandb
from torch.nn import Module

from zoology.model import LanguageModel
from zoology.config import LoggerConfig, TrainConfig

class WandbLogger:
    def __init__(self, config: TrainConfig):
        if config.logger.project_name is None or config.logger.entity is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return
        self.no_logger = False
        self.run = wandb.init(
            name=config.run_id,
            entity=config.logger.entity,
            project=config.logger.project_name, 
        )
        # wandb.run.log_code(
        #     root=str(Path(__file__).parent.parent),
        #     include_fn=lambda path, root: path.endswith(".py")
        # )

    def log_config(self, config: TrainConfig):
        if self.no_logger:
            return
        self.run.config.update(config.model_dump(), allow_val_change=True)

    def log_model(
        self, 
        model: LanguageModel,
        config: TrainConfig
    ):
        if self.no_logger:
            return
        
        max_seq_len = max([c.input_seq_len for c in config.data.test_configs])
        wandb.log(
            {
                "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "state_size": model.state_size(sequence_length=max_seq_len),
            }
        )
        wandb.watch(model)

    def log(self, metrics: dict):
        if self.no_logger:
            return
        wandb.log(metrics)
    
    def finish(self):
        if self.no_logger:
            return
        self.run.finish()


