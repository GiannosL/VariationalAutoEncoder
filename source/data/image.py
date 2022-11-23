import matplotlib.pyplot as plt


class Image:
    def __init__(self, flat_contents, label, 
                 latent_space, dimensions: tuple):
        # image dimensions
        self.x_dims = dimensions[0]
        self.y_dims = dimensions[1]

        # latent space dimensions, tuple
        self.z = latent_space

        # image contents, flat
        self.contents = flat_contents
        self.label = label

        self.n_pixels = flat_contents.shape[0]
    
    def show(self):
        image = self.image_reshape()

        plt.imshow(image)
        plt.title(f"Number: {self.label}")
        plt.axis("off")
        plt.gray()
        plt.show()
    
    def __len__(self):
        return self.contents.shape[0]
    
    def image_reshape(self):
        return self.contents.reshape((self.x_dims, self.y_dims))
