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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='validation')

    def shared_step(self, batch, batch_idx, mode='train'):
        x, label = batch
        out = self(x)

        batch_size, n_class, length = out.size()

        target = (torch.ones((batch_size,)).type_as(label) * label)[:,None].expand(-1, length)
        loss = nn.functional.cross_entropy(out, target)

        with torch.no_grad():
            predictions = (out.max(dim=1)[1] == target)
            acc = predictions.sum() / float(batch_size * length)

        if mode == 'train':
            result = pl.TrainResult(minimize=loss)
        else:
            result = pl.EvalResult(checkpoint_on=loss)

        result.log(f'{mode}_loss', loss)
        result.log(f'{mode}_acc', acc)

        return result