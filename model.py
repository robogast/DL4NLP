import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.wavenet import WaveNetModel

class LanguageModel(pl.LightningModule):
    def __init__(self):
        super(LanguageModel, self).__init__()
        bias = True
        latent_size = 200
        n_classes = 2
        self.training = True
        self.inference_layers = nn.Sequential(
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
                padding=3)
        )
        self.inference_mu = nn.Conv1d(in_channels=500,
                                out_channels=latent_size,
                                bias=bias,
                                padding=2)
        
        self.inference_sigma = nn.Conv1d(in_channels=500,
                                out_channels=latent_size,
                                bias=bias,
                                padding=2)

        self.generative_layers = nn.Sequential(
            nn.Conv1d(in_channels=latent_size,
                out_channels=500,
                kernel_size=9,
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
                kernel_size=3,
                bias=bias,
                padding=2)
        )

        self.cf = nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.Linear(200, n_classes)
        )

    def encode(self, x):
        hidden = nn.ReLU(self.inference_layers(x))
        return self.inference_mu(hidden), nn.Softplus(self.inference_sigma(hidden))

    def decode(self, z):
        return nn.Softplus(self.generative_layers(z))

    def reparameterize(self, mu, var):
        if self.training:
            std = var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, data):
        mu, var = self.encode(data)
        z = self.reparameterize(mu, var)
        return nn.Softmax(self.cf(z)), self.decode(z), mu, var

    def calculate_vae_loss(self, x, reconstruction, mu, var):
        bce = nn.functional.binary_cross_entropy(reconstruction, x)

        kld = -.5 * torch.sum(1 + var - mu.pow(2) - var.exp()) / x.view(-1, input_size).data.shape[0] * input_size

        return bce + kld

    def calculate_cf_loss(self, prediction, label):
        return nn.functional.mse_loss(prediction, label)

    def training_step(self, batch, batch_idx):

        # get inputs and labels from batch
        x, y = batch

        # perform forward pass
        cf_out, reconstruction, mu, var = self.forward(batch)

        # calculate losses
        cf_loss = self.calculate_cf_loss(cf_out, y)
        reconstruction_loss = self.calculate_vae_loss(x, reconstruction, mu, var)
        loss = cf_loss.add(reconstruction_loss)

        return pl.TrainResult(minimize=loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def validation_step(self, batch, batch_idx):

        # get inputs and labels from batch
        x, y = batch

        # perform forward pass
        cf_out, reconstruction, mu, var = self.forward(batch)

        # calculate losses
        cf_loss = self.calculate_cf_loss(cf_out, y)
        reconstruction_loss = self.calculate_vae_loss(x, reconstruction, mu, var)
        loss = cf_loss.add(reconstruction_loss)

        return pl.EvalResult(loss)