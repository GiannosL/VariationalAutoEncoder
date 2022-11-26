import matplotlib.pyplot as plt

def plot_2D_latent(image_list):
    labels_z = {}

    for img in image_list:
        if img.label in labels_z.keys():
            labels_z[img.label].append(img.z)
        else:
            labels_z[img.label] = [img.z]
    
    labels = []
    z_dims = []
    for key in labels_z:
        labels.append(key)
        z_dims.append(labels_z[key])
        #print(key, labels_z[key])
    
    plt.figure(figsize=(10, 7))
    for i in range(len(labels)):
        plt.scatter(x=[j[0] for j in z_dims[i]], 
                    y=[j[1] for j in z_dims[i]], 
                    label=f"{int(labels[i])}",
                    edgecolors="black")
    
    plt.title("Distribution of classes in 2D latent space", weight="bold", fontsize=16)
    plt.xlabel("Latent dimension 1", fontsize=12)
    plt.ylabel("Latent dimension 2", fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
