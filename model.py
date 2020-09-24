import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.wavenet import WaveNetModel

class LanguageModel(pl.LightningModule):
    def __init__(self):
        super(LanguageModel, self).__init__()
        bias = True
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=1,
                out_channels=500,
                kernel_size=3,
                bias=bias,
                padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=500,
                out_channels=500,
                kernel_size=5,
                bias=bias,
                padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=500,
                out_channels=1,
                kernel_size=9,
                bias=bias,
                padding=2)
        )

        self.cf = nn.Sequential(
            nn.Linear(100000, 2)
        )

        self.lossf = nn.CrossEntropyLoss()

    def forward(self, data):
        convs = self.convs(data).squeeze(0)
        out = self.cf(convs)
        return out

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx)
        print(out)
        # assert False, "TODO: training step"
        _, label = batch
        print(label)
        loss = self.lossf(out, label)
        return pl.TrainResult(minimize=loss)
        # return 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx)
        _, label = batch
        print(label)
        print(out.shape, label.shape)
        loss = self.lossf(out, label)
        return pl.EvalResult(loss)

    def shared_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)