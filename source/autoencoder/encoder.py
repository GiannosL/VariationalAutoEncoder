import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dimensions):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dimensions)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)
