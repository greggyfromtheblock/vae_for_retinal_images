"""
Add plotting and introspection functions here
"""
from umap import UMAP
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from utils.training import Encoder
import torch
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import argparse
from torchvision import datasets, transforms
import numpy as np
import torch

from utils.training import Encoder, VAEDataset

import warnings
warnings.filterwarnings("ignore")


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Visualization of encoded/latent vectors of the VAE""")
    parser.add_argument('imfolder', type=str, default=None, metavar='image_dir',
                        help="""The path to the directory which contains the imgge folder. The images themselves must be
                           in one or more subdirectories of the imfolder.""")
    parser.add_argument('csv_file', type=str, default=None, metavar='csv_file',
                        help="""The path to the directory which contain the csv_file (including path and name).""")
    parser.add_argument('network_name', type=str, default=None, metavar='network_name',
                        help="""Only the name of the network.""")
    parser.add_argument('--path_prefix', type=str, default=".", metavar='path_prefix',
                        help="""The path to the directory which should contain the data for tensorboard.""")
    parser.add_argument('--latent_vector_size', type=int, default=32, metavar='latent_vector_size',
                        help="""Size of the latent vector.""")
    args = parser.parse_args()

    imfolder = args.imfolder
    csv_file = args.csv_file
    network_name = args.network_name
    latent_vector_size = args.latent_vector_size
    path_prefix = args.path_prefix

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    RS = 123

    print("Load Data as Tensors and Targets...")
    transform_data = transforms.Compose([transforms.ToTensor(), normalize])
    img_dataset = datasets.ImageFolder(
        imfolder, transform=transform_data
    )
    data = VAEDataset(img_dataset)

    targets =
    # Load network
    trained_encoder = Encoder()
    trained_encoder.load_state_dict(torch.load(f"{path_prefix}/{network_name}/{network_name}" + ".pth"))

    encoded_samples = np.zeros((len(data), latent_vector_size))
    for i, sample in enumerate(data[0]):
        features, _, _ = trained_encoder(sample[0])
        encoded_samples[i] = features.detach().numpy()

    np.save(f"{path_prefix}/{network_name}/encoded_images.npy", encoded_samples)

    # tSNE Visualization of the encoded latent vector
    # Faster if pca is used previously since it reduce the dimensionality
    time_start = time.time()

    pca_5 = PCA(n_components=5)
    pca_result_5 = pca_5.fit_transform(encoded_samples)
    print('\nPCA with 5 components done! Time elapsed: {} seconds'.format(time.time() - time_start))
    print('Cumulative variance explained by 5 principal components: {}'.format(np.sum(pca_5.explained_variance_ratio_)))

    time_start = time.time()
    tsne = TSNE(random_state=RS).fit_transform(encoded_samples)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    tsne_df = pd.DataFrame({'X': tsne[:, 0],
                            'Y': tsne[:, 1],
                            'digit': targets})

    sns.scatterplot(x="X", y="Y",
                    hue="digit",
                    palette=['purple', 'red', 'orange', 'brown', 'blue',
                             'dodgerblue', 'green', 'lightgreen', 'darkcyan', 'black'],
                    legend='full',
                    data=tsne_df);

    plt.title("tSNE-Visualization")
    plt.savefig("./mnist-vae/tSNE_Visualization")
    plt.show()
    plt.close()

    # U-Map Visualization
    clusterable_embedding = UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(encoded_samples)
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
    plt.savefig("./mnist-vae/UMAP-Visualization")
    plt.show()
    plt.close()
