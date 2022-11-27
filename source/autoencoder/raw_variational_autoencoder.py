import torch
import torch.nn as nn

from source.autoencoder.decoder import Decoder
from source.autoencoder.variational_encoder import VariationalEncoder


class RawVariationalAutoEncoder(nn.Module):
    def __init__(self, latent_space_dimensions):
        super(RawVariationalAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU()
        )

        self.z_mean = nn.Linear(512, latent_space_dimensions)
        self.z_log_var = nn.Linear(512, latent_space_dimensions)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_dimensions, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

        self.kl = 0

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterization(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return decoded
    
    def reparameterization(self, z_mu, z_log_var):
        epsilon = torch.randn(1, 2)
        z = z_mu + epsilon * torch.exp(z_log_var/2.)
        return z
    
    def prediction(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        z = self.reparameterization(z_mean, z_log_var)
        decoded = self.decoder(z)
        return decoded, z
