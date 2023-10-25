import argparse
import random
from datetime import datetime
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange
import torchmetrics.functional as tm_f

from zoology.data.utils import prepare_data
from zoology.config import TrainConfig
from zoology.model import LanguageModel
from zoology.logger import WandbLogger


def set_determinism(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def accuracy_ignore_index(logits, y, ignore_index=-100):
    preds = torch.argmax(logits, dim=-1)
    return tm_f.classification.accuracy(
        preds.flatten(),
        y.flatten(),
        num_classes=logits.shape[-1],
        ignore_index=ignore_index,
        average="micro",
        task="multiclass",
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (inputs, targets) in tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader)
        ):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # forward
            logits = self.model(inputs)

            # need to flatten batch and sequence dimensions
            loss = self.loss_fn(
                rearrange(logits, "... c -> (...) c"), targets.flatten()
            )
            loss.backward()
            self.optimizer.step()
            self.logger.log({"train/loss": loss})

    def test(self):
        self.model.eval()
        test_loss = 0
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(
                enumerate(self.test_dataloader), total=len(self.test_dataloader)
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)

                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), targets.flatten()
                )
                test_loss += loss / len(self.test_dataloader)

                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

            all_logits = torch.cat(all_logits, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            test_accuracy = accuracy_ignore_index(all_logits, all_targets)
            print(f"Loss: {test_loss:0.2f} | Acc: {test_accuracy:0.2f}")
            self.logger.log(
                {
                    "valid/loss": test_loss,
                    "valid/accuracy": test_accuracy,
                }
            )

    def fit(self):
        self.model.to("cuda")
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=0.0
        )
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)
            self.test()
            self.scheduler.step()

    def run_profile(self):
        from torch.profiler import profile, record_function, ProfilerActivity

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                self.model(inputs)
            prof.export_chrome_trace("trace.json")
            break
        print("Done!")


def train(config: TrainConfig):
    logger = WandbLogger(config.logger)
    logger.log_config(config)
    config.print()

    train_dataloader, test_dataloader = prepare_data(config.data)
    model = LanguageModel(config=config.model)
    logger.log_model(model)

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )
    task.fit()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train()
