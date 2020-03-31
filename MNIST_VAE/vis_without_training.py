"""
Visualization with PCA and tSNE without previous training.
"""
from umap import UMAP
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mnist_vae import VAEDataset, Encoder, Decoder, MNISTVAETraining
import torch
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patheffects as PathEffects
import pandas as pd
from sklearn.decomposition import PCA


# Utility function to visualize the outputs of PCA and t-SNE
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.show()
    return f, ax, sc, txts


if __name__ == "__main__":
    # Inspired of https://www.datacamp.com/community/tutorials/introduction-t-sne

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    RS = 123

    mnist = MNIST("/home/henrik/PycharmProjects/vae_for_retinal_images/MNIST_VAE/MNIST-Dataset/", download=True, transform=ToTensor())

    # Subset first 20k data points to visualize
    data_size = 1000
    train_data, targets = mnist.data, mnist.targets

    train_data = torch.reshape(train_data[0:data_size], shape=(data_size, 784))
    print(train_data.shape)
    targets = targets[0:data_size].numpy()
    print(np.unique(targets))

    # PCA Visualiziation
    time_start = time.time()

    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(train_data)
    print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))

    pcas= ['pca%i' % i for i in range(10)]
    pca_df = pd.DataFrame(columns=pcas)
    for i, variable in enumerate(pcas):
        pca_df[variable] = pca_result[:, i]

    print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
    top_two_comp = pca_df[['pca1', 'pca2']]  # taking first and second principal component
    fashion_scatter(top_two_comp.values, targets)  # Visualizing the PCA output

    # tSNE Visualization
    # Faster if pca comes previously
    time_start = time.time()

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(train_data)

    print('PCA with 50 components done! Time elapsed: {} seconds'.format(time.time() - time_start))
    print('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

    time_start = time.time()
    fashion_tsne = TSNE(random_state=RS).fit_transform(train_data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # fashion_scatter(fashion_tsne, targets)
    tsne_df = pd.DataFrame({'X': fashion_tsne[:, 0],
                            'Y': fashion_tsne[:, 1],
                            'digit': targets})

    sns.scatterplot(x="X", y="Y",
                    hue="digit",
                    palette=['purple', 'red', 'orange', 'brown', 'blue',
                             'dodgerblue', 'green', 'lightgreen', 'darkcyan', 'black'],
                    legend='full',
                    data=tsne_df);

    plt.title("tSNE-Visualization")
    plt.show()
    plt.close()


    # UMAP-Visualization
    clusterable_embedding = UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(train_data)

    umap_df = pd.DataFrame({'X': clusterable_embedding[:, 0],
                            'Y': clusterable_embedding[:, 1],
                            'digit': targets})

    sns.scatterplot(x="X", y="Y",
                    hue="digit",
                    palette=['purple', 'red', 'orange', 'brown', 'blue',
                             'dodgerblue', 'green', 'lightgreen', 'darkcyan', 'black'],
                    legend='full',
                    data=umap_df);

    # plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=targets, s=0.1, cmap='Spectral');
    plt.title("UMAP-Visualization")
    plt.show()
    plt.close()

