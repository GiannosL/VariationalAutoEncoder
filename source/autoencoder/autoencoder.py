import torch
from source.autoencoder.raw_autoencoder import Raw_Autoencoder
from source.data.image_collection import Image_Collection
from source.data.image import Image

class Autoencoder:
    def __init__(self, latent_dimensions):
        self.model = Raw_Autoencoder(latent_dimensions=latent_dimensions)

    def train(self, data: Image_Collection, n_epochs: int):
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(n_epochs):
            for i in range(0, len(data)):
                X = data[i]

                optimizer.zero_grad()
                # NEED TO MAKE INTO TENSORS
                X_hat = self.model(X)
                loss = ((X - X_hat)**2).sum()
                
                loss.backward()
                optimizer.step()
    
    def transform(self, x, label="Unknown"):
        prediction = self.model.forward(x)
        return Image(prediction.detach().numpy(), label, (28, 28))