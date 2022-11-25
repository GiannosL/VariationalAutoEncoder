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

    def train(self, data: Image_Collection, n_epochs: int, batch_size: int):
        # best optimizer for images is adam
        optimizer = torch.optim.Adam(self.model.parameters())

        # conver images to tensor
        X = self.mini_batch_split(data.images, batch_size)

        for epoch in range(n_epochs):
            if epoch % 2 == 0:
                print(f"Running on epoch: {epoch}")
            
            for element in X:

                optimizer.zero_grad()
                
                X_hat = self.model(element)
                loss = ((element - X_hat)**2).sum()
                
                loss.backward()
                optimizer.step()
        
        self.n_epochs = n_epochs

    def mini_batch_split(self, X, batch_size):
        X = torch.FloatTensor(X)

        # split data in batches
        batches = []
        step = X.shape[0] // batch_size
        for i in range(0, X.shape[0], step):
            batches.append(X[i:i+step, :])
        
        return batches
    
    def transform(self, x, label="Unknown"):
        x = torch.FloatTensor(x)
        
        prediction, latent_space = self.model.prediction(x)
        return Image(prediction.detach().numpy(), label, 
                     dimensions=(28, 28), latent_space=latent_space)
