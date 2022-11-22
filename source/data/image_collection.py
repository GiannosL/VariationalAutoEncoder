import torch
import numpy as np
import matplotlib.pyplot as plt


class Image_Collection:
    def __init__(self, file_path:str, x_dimension:int=None, y_dimension:int=None):
        # collect filepath information
        self.file_path = file_path

        #
        self.x_dim = x_dimension
        self.y_dim = y_dimension

        # read_image file
        self.read_csv()
        self.transform_to_tensors()
    
    def set_image_dimensions(self, x_dimension: int, y_dimension: int):
        self.x_dim = x_dimension
        self.y_dim = y_dimension
    
    def read_csv(self):
        """
        We assume that the label is the first column of the file.
        """
        with open(self.file_path, "r") as f:
            f = f.read()
            f = f.splitlines()
        
        labels = list()
        new_file = list()
        for line in f[1:]:
            new_line = [float(i) for i in line.split(",")]
            labels.append(new_line[0])
            new_file.append(new_line[1:])
        
        self.image_labels = tuple(labels)
        self.image_data = np.array(new_file)
        #
        self.n_images = len(labels)
        self.n_pixels = self.image_data[0].shape[0]
        self.n_labels = len(set(labels))
    
    def transform_to_tensors(self):
        self.image_tensors = torch.FloatTensor(self.image_data)
        self.label_tensors = torch.LongTensor(self.image_labels)
    
    def description(self):
        print("\n---------------------------------------------------------")
        print(f"Number of images: {self.n_images}")
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
