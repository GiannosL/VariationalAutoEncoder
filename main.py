from source.data.image_collection import Image_Collection
from source.autoencoder.autoencoder import Autoencoder


# data input
path_mnist_trainig = "/Users/ljb416/Desktop/projects/datasets/mnist/mnist_train_subset.csv"

#  
dataset = Image_Collection(file_path=path_mnist_trainig, x_dimension=28, y_dimension=28)
dataset.description()
dataset.show_image_pca()

my_autoencoder = Autoencoder(latent_dimensions=10)
my_autoencoder.train(dataset, 20)

denoised_image = my_autoencoder.transform(dataset[0])
denoised_image.show()
