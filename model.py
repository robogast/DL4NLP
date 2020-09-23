import torch
import pytorch_lightning as pl

from models.wavenet import WaveNetModel

class LanguageModel(WaveNetModel):
    def __init__(self):
        super(LanguageModel, self).__init__()

    # def forward(self, data):
        
    #     assert False, "TODO: model forward"

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx)
        print(out)
        # assert False, "TODO: training step"
        # loss = ...
        return pl.TrainResult(minimize=loss)
        # return 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx)
        assert False, "TODO: validation step"
        # loss = ...
        # return pl.ValidationResult(minimize=loss)

    def shared_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)