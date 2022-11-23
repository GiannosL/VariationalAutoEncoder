from source.autoencoder.autoencoder import Autoencoder
from source.data.image_collection import Image_Collection

# data input
path_mnist_training = "/Users/ljb416/Desktop/projects/datasets/mnist/mnist_train_subset.csv"
dataset = Image_Collection(file_path=path_mnist_training, x_dimension=28, y_dimension=28)
dataset.description()

# train autoencoder
my_autoencoder = Autoencoder(latent_dimensions=10)
my_autoencoder.train(dataset, 10)

# make predictions
denoised_image_1 = my_autoencoder.transform(dataset.images[0], dataset.labels[0])
denoised_image_2 = my_autoencoder.transform(dataset.images[1], dataset.labels[1])

# plot data and results
dataset.show_image_pca()
dataset.show_class_distribution()
denoised_image_1.show()
denoised_image_2.show()
