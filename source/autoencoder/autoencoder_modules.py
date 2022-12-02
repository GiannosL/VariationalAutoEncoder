import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dims:int, h1_dims:int, latent_dims:int) -> None:
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, h1_dims)
        self.linear2 = nn.Linear(h1_dims, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, input_dims:int, h1_dims:int, latent_dims:int):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, h1_dims)
        self.linear2 = nn.Linear(h1_dims, input_dims)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class Autoencoder_Model(nn.Module):
    def __init__(self, input_dims:int, h1_dims:int, latent_dims:int):
        super(Autoencoder_Model, self).__init__()
        self.encoder = Encoder(input_dims, h1_dims, latent_dims)
        self.decoder = Decoder(input_dims, h1_dims, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
