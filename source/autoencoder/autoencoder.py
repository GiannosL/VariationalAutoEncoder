import torch
import matplotlib.pyplot as plt

from source.autoencoder.autoencoder import Autoencoder_Model


class AE:
    def __init__(self, n_latent_dimensions) -> None:
        # set latent dimensions for the model
        self.n_latent_dimensions = n_latent_dimensions
        # create the autoencoder
        self.model = AE(latent_dims=self.n_latent_dimensions)
        # set the model's optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def train(self, data, n_epochs:int=15, verbose:bool=True) -> None:
        """
        training the model
        """
        for epoch in range(n_epochs):
            if (epoch % 5 == 0) and (verbose):
                print(f"Epoch {epoch}")
            
            for batch in data:
                self.optimizer.zero_grad()
                x_hat = self.model(batch)

                loss = self.mse(batch, x_hat)
                loss.backward()
                
                self.optimizer.step()
    
    def plot_latent(self, data, y) -> None:
        """
        plot the two dimensional latent space
        """
        for i, x in enumerate(data):
            z = self.model.encoder(x)
            z = z.detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y[i], cmap="tab10", edgecolors="black")
            if i > 100:
                plt.colorbar()
                plt.show()
                break

    def mse(self, real, prediction):
        """
        Calculate Mean Square Error between 
        real data and prediction.
        """
        return ((real - prediction)**2).sum()
