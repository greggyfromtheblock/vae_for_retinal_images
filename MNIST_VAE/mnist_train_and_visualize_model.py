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
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Inspired of https://www.datacamp.com/community/tutorials/introduction-t-sne

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    RS = 123

    mnist = MNIST("/home/henrik/PycharmProjects/vae_for_retinal_images/MNIST_VAE/MNIST-Dataset/", download=True, transform=ToTensor())
    data = VAEDataset(mnist)
    targets = mnist.targets
    # print(len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]), len(data[0][0][0][0]))
    # print(type(data), type(data[0]), type(data[0][0]), type(data[0][0][0]), type(data[0][0][0][0]))


    print("\nStart Training")
    time_start = time.time()
    z = 32
    encoder = Encoder(z)
    decoder = Decoder(z)

    n_epochs = 10
    training = MNISTVAETraining(
    encoder, decoder, data,
        network_name="mnist-vae",
        device="cpu",
        batch_size=128,
        max_epochs=n_epochs,
        verbose=True
    )

    trained_encoder, _ = training.train()
    # print(trained_encoder)
    print('\nTraining with %i epochs done! Time elapsed: %.2f minutes' % (n_epochs, (time.time() - time_start)/60))

    # Save network
    PATH = './mnist_vae.pth'
    torch.save(trained_encoder.state_dict(), PATH)

    # Load network
    # In this case unnecessary, but so we would be able to load the trained network in another script
    trained_encoder = Encoder(z)
    trained_encoder.load_state_dict(torch.load(PATH))

    encoded_samples = np.zeros((len(data), z))
    for i, sample in enumerate(data[0]):
            features, _, _ = trained_encoder(sample[0])
            encoded_samples[i] = features.detach().numpy()

    np.save(f"./mnist-vae/features.npy", encoded_samples)

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
