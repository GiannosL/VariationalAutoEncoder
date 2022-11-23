import torch
import matplotlib.pyplot as plt

from source.data.image import Image
from source.data.image_collection import Image_Collection
from source.autoencoder.raw_autoencoder import Raw_Autoencoder


class Autoencoder:
    def __init__(self, latent_dimensions):
        self.model = Raw_Autoencoder(latent_dimensions=latent_dimensions)
        self.training_loss = None
        self.n_epochs = None

    def train(self, data: Image_Collection, n_epochs: int):
        training_loss = []
        optimizer = torch.optim.Adam(self.model.parameters())
        X = torch.FloatTensor(data.images)

        for epoch in range(n_epochs):

            optimizer.zero_grad()
            
            X_hat = self.model(X)
            loss = ((X - X_hat)**2).sum()
            training_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        self.n_epochs = n_epochs
        self.training_loss = tuple(training_loss)
    
    def transform(self, x, label="Unknown"):
        x = torch.FloatTensor(x)
        prediction = self.model.forward(x)
        return Image(prediction.detach().numpy(), label, (28, 28))
    
    def show_loss_trajectory(self):
        plt.plot(range(1, self.n_epochs+1), self.training_loss, color="black")
        plt.title("Loss trajectory during training", weight="bold", fontsize=16)
        plt.xlabel("Number of epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.show()
