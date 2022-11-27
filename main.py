from source.data.plot_latent_space import plot_2D_latent
from source.autoencoder.autoencoder import Autoencoder
from source.data.image_collection import Image_Collection
from source.autoencoder.variational_autoencoder import VariationalAutoEncoder

# data input
path_mnist_training = "/Users/ljb416/Desktop/projects/datasets/mnist/mnist_train.csv"
dataset = Image_Collection(file_path=path_mnist_training, x_dimension=28, y_dimension=28)
dataset.description()

vae = True

if not vae:
    # train autoencoder
    my_autoencoder = Autoencoder(latent_dimensions=2)
    my_autoencoder.train(data=dataset, n_epochs=10, batch_size=50)

    # make predictions
    images_list = []
    for i in range(30):
        img = my_autoencoder.transform(x=dataset.images[i], label=dataset.labels[i])
        images_list.append(img)

    # plot data and results
    dataset.show_image_pca()
    dataset.show_class_distribution()
    plot_2D_latent(image_list=images_list)

else:
    import numpy as np
    # variational autoencoder
    print("\n Here starts the Variational Autoencoder")
    my_variational_autoencoder = VariationalAutoEncoder(latent_dimensions=2)
    my_variational_autoencoder.train(data=dataset, epochs=10, batch_size=50)

    # make predictions
    images_list = []
    for i in range(1):
        img = my_variational_autoencoder.transform(x=dataset.images[i], label=dataset.labels[i])
        print(img.z)
        images_list.append(img)
    
    exit()
    # plot data and results
    dataset.show_image_pca()
    dataset.show_class_distribution()
    plot_2D_latent(image_list=images_list)
