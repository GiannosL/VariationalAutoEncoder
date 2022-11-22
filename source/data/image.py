import numpy as np
import matplotlib.pyplot as plt


class Image:
    def __init__(self, flat_image, x_dimensions, y_dimensions):
        self.flat_image = flat_image
        self.x_dims = x_dimensions
        self.y_dims = y_dimensions
    
    def show(self):
        image = self.image_reshape()

        plt.imshow(image)
        plt.title(f"Number: 5")
        plt.axis("off")
        plt.gray()
        plt.show()
    
    def image_reshape(self):
        return self.flat_image.reshape((self.x_dims, self.y_dims))
