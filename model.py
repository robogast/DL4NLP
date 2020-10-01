import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.wavenet import WaveNetModel
from pytorch_lightning.metrics.classification import Accuracy

class LanguageModel(WaveNetModel):
    def __init__(self, *args, **kwargs):
        super(LanguageModel, self).__init__(*args, **kwargs)
        self.lossf = nn.CrossEntropyLoss()
        self.metric = Accuracy(self.outclasses)

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx)
        _, label = batch
        loss = self.lossf(out, label)

        acc = self.metric(out.detach().max(1)[1], label)
        
        train_result = pl.TrainResult(minimize=loss)
        train_result.log('train_loss', loss)
        train_result.log('train_acc', acc)
        return train_result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx)
        _, label = batch
        loss = self.lossf(out, label)
        acc = self.metric(out.max(1)[1], label)

        validation_result = pl.EvalResult(loss)
        validation_result.log('validation_loss', loss)
        validation_result.log('validation_acc', acc)
        return validation_result

    def shared_step(self, batch, batch_idx):
        x, _ = batch
        out = self(x)
        return out