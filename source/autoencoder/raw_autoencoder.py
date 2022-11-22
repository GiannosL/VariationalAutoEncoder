import torch
import torch.nn as nn

from source.autoencoder.encoder import Encoder
from source.autoencoder.decoder import Decoder

class Raw_Autoencoder(nn.Module):
    def __init__(self, latent_dimensions):
        super(Raw_Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dimensions=latent_dimensions)
        self.decoder = Decoder(latent_dimensions=latent_dimensions)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

