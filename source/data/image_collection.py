import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from source.data.image import Image

class Image_Collection:
    def __init__(self, file_path:str, x_dimension:int, y_dimension:int):
        # collect filepath information
        self.file_path = file_path

        # image dimensions
        self.x_dim = x_dimension
        self.y_dim = y_dimension

        # read_image file
        self.read_csv()
    
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
            #images.append( Image(torch.FloatTensor(new_line[1:]), label=torch.LongTensor([new_line[0]]), dimensions=(self.x_dim, self.y_dim)) )
            labels.append(new_line[0])
            # this is used for the pca data structure
            images.append(new_line[1:])
        
        self.labels = labels
        self.images = np.array(images)
        self.n_labels = len(set(labels))
        self.n_pixels = self.images[0].shape[0]

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
    
    def show_image_pca(self):
        pca_model = PCA(n_components=self.n_labels)
        # do standarization
        pcs = pca_model.fit_transform(self.images)
        eigenvalue_ratio = pca_model.explained_variance_ratio_

        pc_dict = {}
        for i in range(pcs.shape[0]):
            if self.labels[i] in pc_dict.keys():
                pc_dict[self.labels[i]].append(pcs[i, :])
            else:
                pc_dict[self.labels[i]] = [pcs[i, :]]
        
        pc_dict = {i: np.array(pc_dict[i]) for i in pc_dict}

        plt.figure(figsize=(10, 7))
        for i in pc_dict:
            plt.scatter(pc_dict[i][:, 0], pc_dict[i][:, 1], label=str(i), edgecolors="black")
        
        plt.title("PCA visualization")
        plt.xlabel(f"PC-1, {eigenvalue_ratio[0]*100:.2f}%")
        plt.ylabel(f"PC-2, {eigenvalue_ratio[1]*100:.2f}%")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def __len__(self):
        return len(self.images)
