import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dimensions):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dimensions, 512)
        self.linear2 = nn.Linear(512, 784)
    
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z
