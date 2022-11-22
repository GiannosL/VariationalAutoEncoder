import torch
import numpy as np
import matplotlib.pyplot as plt

from source.data.image import Image

class Image_Collection:
    def __init__(self, file_path:str, x_dimension:int, y_dimension:int):
        # collect filepath information
        self.file_path = file_path

        #
        self.x_dim = x_dimension
        self.y_dim = y_dimension

        # read_image file
        self.read_csv()

        #
        self.n_pixels = self.images[0].n_pixels
    
    def read_csv(self):
        """
        We assume that the label is the first column of the file.
        """
        with open(self.file_path, "r") as f:
            f = f.read()
            f = f.splitlines()
        
        images = []
        labels = []
        for line in f[1:]:
            new_line = [float(i) for i in line.split(",")]
            images.append(
                Image(torch.FloatTensor(new_line[1:]), 
                      label=torch.LongTensor([new_line[0]]), 
                      dimensions=(self.x_dim, self.y_dim))
                )
            labels.append(new_line[0])
        
        self.images = images
        self.n_labels = len(set(labels))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, img_index):
        return self.images[img_index].contents

    def description(self):
        print("\n---------------------------------------------------------")
        print(f"Number of images: {self.__len__()}")
        print(f"Number of pixels per image: {self.n_pixels}")
        print(f"Number of unique labels: {self.n_labels}")
        print("---------------------------------------------------------\n")

    def show_image(self, image_index: int):
        if not self.y_dim:
            raise Exception("You have to set image dimensions first")
        
        image = self.image_data[image_index].reshape((self.x_dim, self.y_dim))
        label = self.image_labels[image_index]

        #
        plt.imshow(image)
        plt.title(f"Number: {label}", weight="bold", fontsize=16)
        plt.axis("off")
        plt.gray()
        plt.show()
