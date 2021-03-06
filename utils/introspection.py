"""
Add plotting and introspection functions here
"""
import argparse
import os
import time
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from training import Encoder, VAEDataset
from umap import UMAP
from utils import setup

warnings.filterwarnings("ignore")


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def add_slash(path):
    if path[-1] != '/':
        return path + "/"
    else:
        return(path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Introspection""")
    parser.add_argument('imdir', type=str, default=None, metavar='image_dir',
                        help="""The path to the directory which contains the preprocessed image folder.""")
    parser.add_argument('csv_file', type=str, default=None, metavar='csv_file',
                        help="""Processed annotations fileeeee""")

    args, rest = parser.parse_known_args()
    imfolder = add_slash(os.path.abspath(args.imdir))
    csv_file = os.path.abspath(args.csv_file)

    FLAGS, logger = setup(running_script="./utils/introspection.py",args=rest, config="config.json")
    print("FLAGS= ", FLAGS)

    latent_vector_size = FLAGS.zdim

    network_name = FLAGS.networkname
    path_prefix = FLAGS.path_prefix
    network_dir = f'{path_prefix}/{network_name}/'

    print("\nLoad Data as Tensors...")
    transform_data = transforms.Compose([transforms.ToTensor(), normalize])
    img_dataset = datasets.ImageFolder(
        imfolder, transform=transform_data
    )
    data = VAEDataset(img_dataset)
    print("\nSize of the dataset: {}\nShape of the single tensors: {}".format(len(data), data[0][0].shape))

    print("\nBuild targets...")
    csv_df = pd.read_csv(csv_file, sep='\t')

    diagnoses = {
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
    number_of_diagnoses = len(diagnoses)  # not sure if others should be an own category
    data_size = len(data)
    targets = np.zeros((data_size, number_of_diagnoses),  dtype=np.int8)

    angles = [x for x in range(-FLAGS.max_degree, -9)]
    angles.extend([x for x in range(10, FLAGS.max_degree+1)])
    angles.extend([x for x in range(-9, 9)])
    print("\nPossible Angles: {}\n".format(angles))

    for i, jpg in tqdm(enumerate(os.listdir(imfolder+"images"))):
        jpg = jpg.replace("_flipped", "")

        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]
        for j, feature in enumerate(diagnoses.keys()):
            targets[i][j] = csv_df.iloc[row_number].at[feature]

    print("Finished building targets...")

    # Load network
    trained_encoder = Encoder()
    trained_encoder.load_state_dict(torch.load(network_dir+f"{network_name}.pth"))

    print("Generate samples..")
    samples = torch.zeros((data_size, *data[0][0].shape))
    encoded_samples = np.zeros((data_size, latent_vector_size))
    for i in tqdm(range(0, data_size, data_size)):
        samples[i] = data[i][0]

    print("\nStart encoding of each image...")
    features, _, _ = trained_encoder(samples)
    encoded_samples = features.detach().numpy()
    print("Finished encoding of each image...")

    os.makedirs(network_dir+"/Visualizations/", exist_ok=True)
    print("Start Visualization...")
    colormap = np.array(['darkorange', 'royalblue'])

    for i, diagnosis in tqdm(enumerate(diagnoses.keys())):

        # tSNE Visualization of the encoded latent vector
        time_start = time.time()
        tsne = TSNE(random_state=123).fit_transform(encoded_samples)

        # U-Map Visualization
        clusterable_embedding = UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=2,
            random_state=42,
        ).fit_transform(encoded_samples)

        orange_patch = mpatches.Patch(color=colormap[0], label=f'No {diagnoses[diagnosis]}')
        blue_patch = mpatches.Patch(color=colormap[1], label=f'{diagnoses[diagnosis]}')
        diagnosis_name = diagnoses[diagnosis]

        plt.scatter(tsne[:, 0], tsne[:, 1], c=colormap[targets[:, i]], s=1)
        plt.legend(handles=[orange_patch, blue_patch])
        plt.title(f"tSNE-Visualization of diagnosis: {diagnosis_name}\n", fontsize=16, fontweight='bold')

        plt.savefig(f"{path_prefix}/{network_name}/Visualizations/tsne_visualization_of_diagnosis_{diagnosis_name}.png")
        plt.show()
        plt.close()

        plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=colormap[targets[:, i]], s=1, label=colormap)
        plt.legend(handles=[orange_patch, blue_patch])
        plt.title(f"UMAP-Visualization of diagnosis: {diagnosis_name}\n", fontsize=16, fontweight='bold')

        plt.savefig(f"{path_prefix}/{network_name}/Visualizations/umap_visualization_of_diagnosis_{diagnosis_name}.png")
        plt.show()
        plt.close()

        tsne_df = pd.DataFrame({'X': tsne[:, 0],
                                'Y': tsne[:, 1],
                                f'{diagnoses[diagnosis]}': targets[:, i]})

        umap_df = pd.DataFrame({'X': clusterable_embedding[:, 0],
                                'Y': clusterable_embedding[:, 1],
                                diagnoses[diagnosis]: targets[:, i]})
