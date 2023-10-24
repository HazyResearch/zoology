import argparse
import random
from datetime import datetime
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
from einops import rearrange
import torchmetrics.functional as tm_f

from zoology.data.utils import prepare_data
from zoology.config import TrainConfig
from zoology.model import LanguageModel


def set_determinism(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def accuracy_ignore_index(logits, y, ignore_index=-100):
    num_classes = logits.shape[-1]
    preds = torch.argmax(logits, dim=-1)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return tm_f.classification.accuracy(preds, y, num_classes=num_classes, ignore_index=ignore_index, average='micro', task='multiclass')


class Trainer:
    def __init__(
        self,   
        model: nn.Module, 
        train_dataloader: DataLoader =None,
        test_dataloader: DataLoader=None,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        device: Union[str, int] = "cuda"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.device = device
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

    def train_epoch(self, epoch):
        self.model.train()
        all_outputs = []
        all_targets = []
        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # forward
            outputs = self.model(inputs)

            # loss computation
            outputs = outputs[0]
            outputs = rearrange(outputs, '... C -> (...) C')
            targets = rearrange(targets, '... -> (...)')
            all_outputs.append(outputs)
            all_targets.append(targets)
            
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

        # backprop
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        loss = self.loss_fn(all_outputs, all_targets)
        print(f"Epoch: {epoch} | Loss: {loss.item()}")
        
    def test(self):
        self.model.eval()
        test_loss = 0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs[0]
                outputs = rearrange(outputs, '... C -> (...) C')
                targets = rearrange(targets, '... -> (...)')
                all_outputs.append(outputs)
                all_targets.append(targets)
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            sample_size = all_outputs.shape[0] // inputs.shape[1]
            loss = self.loss_fn(all_outputs, all_targets)
            test_loss += loss.item()
            test_accuracy = accuracy_ignore_index(all_outputs, all_targets)
            print((batch_idx, len(self.test_dataloader), 'Loss: %.3f | Acc: %.3f%% | Samples: %d' % ((test_loss/(batch_idx+1)), test_accuracy, sample_size)))

    def fit(self):
        self.model.to('cuda')
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=0.0)
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)
            self.test()
            self.scheduler.step()

    def run_profile(self):
        from torch.profiler import profile, record_function, ProfilerActivity
        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                self.model(inputs)
            prof.export_chrome_trace("trace.json")
            break
        print("Done!")
            

        
def main():
    config = TrainConfig.from_cli()

    train_dataloader, test_dataloader = prepare_data(config.data)
    model = LanguageModel(config=config.model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        device=device
    )
    task.fit()    

if __name__ == "__main__":
    main()