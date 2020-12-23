import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

num_samples = 10000

train = RandomDataset(32, num_samples)
train = DataLoader(train, batch_size=32)

val = RandomDataset(32, num_samples)
val = DataLoader(val, batch_size=32)

test = RandomDataset(32, num_samples)
test = DataLoader(test, batch_size=32)

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x['x'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('fake_test_acc', loss)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def test_gpus(gpus):
    model = BoringModel()

        #progress_bar_refresh_rate=20
    #trainer = pl.Trainer(num_sanity_val_steps=0, max_epochs=1, progress_bar_refresh_rate=20)

    trainer = pl.Trainer(gpus=gpus, distributed_backend='ddp',
                         num_sanity_val_steps=0, max_epochs=1)
                         

    trainer.fit(model, train, val)

    trainer.test(test_dataloaders=test)


if __name__ == '__main__':
    gpus = sys.argv[1]
    test_gpus(gpus)
