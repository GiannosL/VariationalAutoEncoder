import torch
import torch.nn as nn
import torch.nn.functional as F

from source.autoencoder.autoencoder_modules import Decoder


class VariationalEncoder(nn.Module):
    def __init__(self, input_dims:int, h1_dims:int, latent_dims:int) -> None:
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, h1_dims)
        self.linear2 = nn.Linear(h1_dims, latent_dims)
        self.linear3 = nn.Linear(h1_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class Variational_Autoencoder_Model(nn.Module):
    def __init__(self, input_dims:int, h1_dims:int, latent_dims:int) -> None:
        super(Variational_Autoencoder_Model, self).__init__()
        self.encoder = VariationalEncoder(input_dims, h1_dims, latent_dims)
        self.decoder = Decoder(input_dims, h1_dims, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
