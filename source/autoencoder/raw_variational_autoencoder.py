import torch.nn as nn

from decoder import Decoder
from variational_encoder import VariationalEncoder


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_space_dimensions):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = VariationalEncoder(latent_space_dimensions)
        self.decoder = Decoder(latent_space_dimensions)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def predict(self, x):
        z = self.encoder(x)
        return self.decoder(z), z.detach().numpy()
