import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.wavenet import WaveNetModel

class LanguageModel(WaveNetModel):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.lossf = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx)
        # print(out)
        # assert False, "TODO: training step"
        _, label = batch
        # print(label)
        loss = self.lossf(out, label)
        print(out.shape, label.shape)
        return pl.TrainResult(minimize=loss)
        # return 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx)
        _, label = batch
        loss = self.lossf(out, label)
        return pl.EvalResult(loss)

    def shared_step(self, batch, batch_idx):
        x, _ = batch
        out = self(x)
        return out