"""
Add plotting and introspection functions here
"""
from umap import UMAP
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import datasets, transforms
import numpy as np
import torch
import os
from tqdm import tqdm
from training import Encoder, VAEDataset
from utils import setup
import warnings
warnings.filterwarnings("ignore")


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


if __name__ == '__main__':

    FLAGS, logger = setup(running_script="./utils/introspection.py", config="config.json")
    print("FLAGS= ", FLAGS)

    imfolder = os.path.abspath(FLAGS.input)
    csv_file = FLAGS.input_csv
    network_name = FLAGS.networkname
    latent_vector_size = FLAGS.zdim
    path_prefix = FLAGS.path_prefix

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    RS = 123

    print("\nLoad Data as Tensors...")
    transform_data = transforms.Compose([transforms.ToTensor(), normalize])
    img_dataset = datasets.ImageFolder(
        imfolder, transform=transform_data
    )
    data = VAEDataset(img_dataset)
    print("\nSize of the dataset: {}\nShape of the single tensors: {}".format(len(data), data[0][0].shape))

    print("\nBuild targets...")
    csv_df = pd.read_csv(csv_file, sep='\t')
    # print(csv_df['Fundus Image'])
    # print(list(csv_df.columns))
    features = {
        "N": "normal fundus",
        "D": "proliferative retinopathy",
        "G": "glaucoma",
        "C": "cataract",
        "A": "age related macular degeneration",
        "H": "hypertensive retinopathy",
        "M": "myopia",
        # "ant": "anterior segment",
        # "no": "no fundus image",
    }
    number_of_diagnoses = len(features)  # not sure if others should be an own category
    targets = np.zeros((len(data), number_of_diagnoses))

    for i, jpg in tqdm(enumerate(os.listdir(imfolder+"/train/"))):
        jpg = jpg.replace("_flipped", "")

        for angle in range(-FLAGS.max_degree, FLAGS.max_degree+1):
            if angle != 0:
                jpg = jpg.replace("_rot_%i" % angle, "")
        print(jpg)
        row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]
        for j, feature in enumerate(features.keys()):
            targets[i][j] = csv_df.iloc[row_number].at[feature]

    print("Finished building targets...")

    # Load network
    trained_encoder = Encoder()
    trained_encoder.load_state_dict(torch.load(f"{path_prefix}/{network_name}/{network_name}" + ".pth"))

    print("\nStart encoding of each image...")
    encoded_samples = np.zeros((len(data), latent_vector_size))
    for i, sample in tqdm(enumerate(data[0])):
        features, _, _ = trained_encoder(sample[0])
        encoded_samples[i] = features.detach().numpy()
    print("Finished encoding of each image...")

    print("Start Visualization...")
    for i, feature in tqdm(enumerate(features.keys())):

        # tSNE Visualization of the encoded latent vector
        time_start = time.time()
        tsne = TSNE(random_state=RS).fit_transform(encoded_samples)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        tsne_df = pd.DataFrame({'X': tsne[:, 0],
                                'Y': tsne[:, 1],
                                feature: np.transpose(targets[i])})

        # U-Map Visualization
        clusterable_embedding = UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(encoded_samples)
        umap_df = pd.DataFrame({'X': clusterable_embedding[:, 0],
                                'Y': clusterable_embedding[:, 1],
                                feature: np.transpose(targets[i])})

        plt.subplot(2, 1, 1)
        sns.scatterplot(x="X", y="Y",
                        hue=feature,
                        palette=['orange', 'blue'],
                        legend='full',
                        data=tsne_df);

        plt.title("tSNE-Visualization")

        plt.subplot(2, 1, 2)
        sns.scatterplot(x="X", y="Y",
                        hue=feature,
                        palette=['purple', 'red', 'orange', 'brown', 'blue',
                                 'dodgerblue', 'green', 'lightgreen', 'darkcyan', 'black'],
                        legend='full',
                        data=umap_df);

        # plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=targets, s=0.1, cmap='Spectral');
        plt.title("UMAP-Visualization")

        plt.savefig(f"{path_prefix}/{network_name}/Visualization")
        plt.show()
        plt.close()


    """
     csv_df = pd.read_csv(path, sep='\t')
    >>> a=list(csv_df.columns)
    >>> a
    ['ID', 'Side', 'Patient Age', 'Patient Sex', 'Fundus Image', 'Diagnostic Keywords', 'N',
     'D', 'G', 'C', 'A', 'H', 'M', 'O', 'anterior', 'no fundus']
    >>> d=csv_df.'ID'
      File "<stdin>", line 1
        d=csv_df.'ID'
                    ^
    SyntaxError: invalid syntax
    >>> d=csv_df['ID']
    >>> d=csv_df['Fundus Image']
    >>> d
    0           0_left.jpg
    1          0_right.jpg
    2           1_left.jpg
    3          1_right.jpg
    4           2_left.jpg
                 ...      
    6995    4689_right.jpg
    6996     4690_left.jpg
    6997    4690_right.jpg
    6998     4784_left.jpg
    6999    4784_right.jpg
    Name: Fundus Image, Length: 7000, dtype: object
    >>> 
    """