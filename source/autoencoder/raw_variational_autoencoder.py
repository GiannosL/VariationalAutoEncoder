import torch.nn as nn

from source.autoencoder.decoder import Decoder
from source.autoencoder.variational_encoder import VariationalEncoder


class RawVariationalAutoEncoder(nn.Module):
    def __init__(self, latent_space_dimensions):
        super(RawVariationalAutoEncoder, self).__init__()

        self.encoder = VariationalEncoder(latent_space_dimensions)
        self.decoder = Decoder(latent_space_dimensions)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def prediction(self, x):
        z = self.encoder(x)
        return self.decoder(z), z.detach().numpy()
