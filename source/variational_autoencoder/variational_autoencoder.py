import torch
import matplotlib.pyplot as plt

from source.variational_autoencoder.variational_autoencoder_modules import Variational_Autoencoder_Model


class VAE:
    def __init__(self, n_latent_dimensions:int) -> None:
        self.n_latent_dimensions = n_latent_dimensions

        self.model = Variational_Autoencoder_Model(latent_dims=self.n_latent_dimensions)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, data, epochs:int=20, verbose:bool=True) -> None:
        for epoch in range(epochs):
            if (epoch % 5 == 0) and (verbose):
                print(f"Epoch {epoch}")
            
            for batch, _ in data:
                self.optimizer.zero_grad()
                x_hat = self.model(batch)
                loss = self.mse(batch, x_hat) + self.model.encoder.kl
                loss.backward()
                self.optimizer.step()
        
    def mse(self, real, prediction):
        return ((real - prediction)**2).sum()
    
    def transform(self, x):
        return self.model(x)

    def plot_latent(self, data, num_batches=100) -> None:
        plt.figure(figsize=(10, 7))
        
        for i, (batch, labels) in enumerate(data):
            z = self.model.encoder(batch)
            z = z.detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', edgecolors="black")
            if i > num_batches:
                plt.colorbar()
                plt.title("Latent space depiction", weight="bold", fontsize=16)
                plt.xlabel("Dimension 1", fontsize=12)
                plt.ylabel("Dimension 2", fontsize=12)
                plt.show()
                break
