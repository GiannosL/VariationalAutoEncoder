#!/usr/bin/env python3
import torchvision
import torch._utils

from source.autoencoder.autoencoder import AE
from source.variational_autoencoder.variational_autoencoder import VAE

# load the input dataset
dataset = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True), batch_size=128, shuffle=True)

# train the autoencoder model
print("Training AutoEncoder:")
the_autoencoder = AE(n_latent_dimensions=2)
the_autoencoder.train(data=dataset, n_epochs=10, verbose=True)

#
print("\nTraining Variational AutoEncoder:")
the_variational_autoencoder = VAE(n_latent_dimensions=2)
the_variational_autoencoder.train(data=dataset, epochs=10)

# plot 2D versions of the latent space for AE and VAE
the_autoencoder.plot_latent(data=dataset)
the_variational_autoencoder.plot_latent(data=dataset)
