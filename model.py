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

    def calculate_var(self, batch, batch_idx):
        x, _ = batch
        n_mc = 5
        with torch.no_grad():
            sum_x, sum_x_squared = 0, 0
            for _ in range(n_mc):
                out = self(x)
                sum_x += out
                sum_x_squared += out ** 2

            var = (1 / n_mc) * (sum_x_squared - (sum_x ** 2) / n_mc)

        return var

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='validation')

    def test_step(self, batch, batch_idx):
        var = self.calculate_var(batch, batch_idx)


    def shared_step(self, batch, batch_idx, mode='train'):
        x, label = batch
        out = self(x)

        batch_size, n_class, length = out.size()

        target = (torch.ones((batch_size,)).type_as(label) * label)[:,None].expand(-1, length)
        loss = nn.functional.cross_entropy(out, target)

        with torch.no_grad():
            acc = (out.max(dim=1)[1] == target).sum() / float(batch_size * length)

        if mode == 'train':
            result = pl.TrainResult(minimize=loss)
        else:
            result = pl.EvalResult(checkpoint_on=loss)
            # var = self.calculate_var(batch, batch_idx).mean(dim=2)

            result.log(f'{mode}_var', loss)

        result.log(f'{mode}_loss', loss)
        result.log(f'{mode}_acc', acc)

        return result