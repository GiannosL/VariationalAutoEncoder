import torch

from source.data.image import Image
from source.data.image_collection import Image_Collection
from source.autoencoder.raw_variational_autoencoder import VariationalAutoEncoder


class Variational_AutoEncoder:
    def __init__(self, latent_dimensions):
        self.model = VariationalAutoEncoder(latent_space_dimensions=latent_dimensions)
    
    def train(self, data: Image_Collection, epochs: int, batch_size: int):
        #
        opt = torch.optim.Adam(self.model.parameters())
        data = self.mini_batch_split(data, batch_size)

        for epoch in range(epochs):
            for x in data:
                opt.zero_grad()
                x_hat = self.model(x)
                loss = ((x - x_hat)**2).sum() + self.model.encoder.kl
                loss.backward()
                opt.step()

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
