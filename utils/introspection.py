"""
Add plotting and introspection functions here
"""
from umap import UMAP
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def add_slash(path):
    if path[-1] != '/':
        return path + "/"
    else:
        return(path)


if __name__ == '__main__':

    FLAGS, logger = setup(running_script="./utils/introspection.py", config="config.json")
    print("FLAGS= ", FLAGS)

    imfolder = add_slash(os.path.abspath(FLAGS.input))
    csv_file = os.path.abspath(FLAGS.csv_input)
    latent_vector_size = FLAGS.zdim

    network_name = FLAGS.network_name
    path_prefix = FLAGS.path_prefix
    network_dir = os.path.abspath(f'{path_prefix}/{network_name}/')

    print("\nLoad Data as Tensors...")
    img_dataset = datasets.ImageFolder(
        os.path.dirname(os.path.dirname(imfolder)),
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )
    data = VAEDataset(img_dataset)
    print("\nSize of the dataset: {}\nShape of the single tensors: {}".format(len(data), data[0][0].shape))

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
    number_of_diagnoses = len(diagnoses)
    data_size = len(data)

    targets = np.zeros((data_size, number_of_diagnoses+1),  dtype=np.uint8)
    age_targets = np.zeros(data_size,  dtype=np.uint8)

    angles = [x for x in range(-FLAGS.maxdegree, -9)]
    angles.extend([x for x in range(10, FLAGS.maxdegree+1)])
    angles.extend([x for x in range(-9, 10)])
    print("\nPossible Angles: {}\n".format(angles))

    print("\nBuild targets...")
    marker = None
    for i, jpg in tqdm(enumerate(os.listdir(imfolder))):
        if jpg == '.snakemake_timestamp':
            marker = True
            continue
        jpg = jpg.replace("_flipped", "")

        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]

        diagnoses_list = list(diagnoses.keys())
        diagnoses_list.extend(["Patient Sex"])
        for j, feature in enumerate(diagnoses_list):
            if not marker:
                if feature == "N":
                    targets[i][j] = not csv_df.iloc[row_number].at[feature]
                else:
                    if feature == "Patient Sex":
                        targets[i][j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                    else:
                        targets[i][j] = csv_df.iloc[row_number].at[feature]
            else:
                if feature == "N":
                    targets[i-1][j] = not csv_df.iloc[row_number].at[feature]
                else:
                    if feature == "Patient Sex":
                        targets[i-1][j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                    else:
                        targets[i-1][j] = csv_df.iloc[row_number].at[feature]
        if marker:
            age_targets[i-1] = csv_df.iloc[row_number].at["Patient Age"]
        else:
            age_targets[i-1] = csv_df.iloc[row_number].at["Patient Age"]

    # Load network
    trained_encoder = Encoder()
    # trained_encoder.load_state_dict(torch.load(network_dir+f"/{network_name}.pth"))

    print("Generate samples..")
    def calc_batch_size(batch_size=8):
        global data_size
        if data_size % batch_size == 0:
            return batch_size
        if batch_size == 3:
            return batch_size
        else:
            return calc_batch_size(batch_size-1)

    # calculate batch_size
    batch_size = calc_batch_size(batch_size=8)
    samples = torch.zeros((batch_size, *data[0][0].shape))
    d_mod_b = data_size % batch_size
    encoded_samples = np.zeros((data_size, latent_vector_size))

    for i in tqdm(range(0, data_size, batch_size)):
        if (i + batch_size) < data_size:
            for j in range(batch_size):
                samples[j] = data[i+j][0]
            features, _, _ = trained_encoder(samples)
            encoded_samples[i:(i+batch_size)] = features.detach().numpy()
        else:
            # for uncompleted last batch
            samples = torch.zeros((d_mod_b, *data[0][0].shape))
            for i in range(data_size-d_mod_b, data_size, d_mod_b):
                for j in range(d_mod_b):
                    samples[j] = data[i+j][0]
                features, _, _ = trained_encoder(samples)
                encoded_samples[i:(i + d_mod_b)] = features.detach().numpy()
    print("Finished encoding of each image...")

    # tSNE Visualization of the encoded latent vector
    tsne = TSNE(random_state=123).fit_transform(encoded_samples)

    # U-Map Visualization
    clusterable_embedding = UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(encoded_samples)

    os.makedirs(f'{network_dir}/visualizations/', exist_ok = True)
    print("\nStart Visualization...")
    for i, diagnosis in tqdm(enumerate(diagnoses_list)):
        if diagnosis != "Patient Sex":
            colormap = np.array(['g', 'r'])
            diagnosis_name = diagnoses[diagnosis]
            if diagnosis != "N":
                red_patch = mpatches.Patch(color=colormap[0], label=f'no {diagnosis_name}')
                green_patch = mpatches.Patch(color=colormap[1], label=f' {diagnosis_name}')
            else:
                red_patch = mpatches.Patch(color=colormap[0], label=f'{diagnosis_name}')
                green_patch = mpatches.Patch(color=colormap[1], label=f'no {diagnosis_name}')
        else:
            colormap = np.array(['darkorange', 'royalblue'])
            orange_patch = mpatches.Patch(color=colormap[0], label=f'Female')
            blue_patch = mpatches.Patch(color=colormap[1], label=f'Male')
            diagnosis_name = "Patient Sex"

        plt.scatter(tsne[:, 0], tsne[:, 1], c=colormap[targets[:, i]], s=1)
        if diagnosis != "Patient Sex":
            plt.legend(handles=[red_patch, green_patch])
        else:
            plt.legend(handles=[orange_patch, blue_patch])

        plt.title(f"tSNE-Visualization of diagnosis: {diagnosis_name}\n", fontsize=16, fontweight='bold')

        plt.savefig(
            f"{path_prefix}/{network_name}/visualizations/tsne_visualization_of_diagnosis_{diagnosis_name}.png")
        plt.show()
        plt.close()

        plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=colormap[targets[:, i]], s=1,
                    label=colormap)

        if diagnosis != "Patient Sex":
            plt.legend(handles=[red_patch, green_patch])
        else:
            plt.legend(handles=[orange_patch, blue_patch])

        plt.title(f"UMAP-Visualization of diagnosis: {diagnosis_name}\n", fontsize=16, fontweight='bold')

        plt.savefig(
            f"{path_prefix}/{network_name}/visualizations/umap_visualization_of_diagnosis_{diagnosis_name}.png")
        plt.show()
        plt.close()

    # Plot feature "Patient Age"
    plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], s=1, c=age_targets)
    plt.title("Point observations")
    cbar = plt.colorbar()
    cbar.set_label("Age", labelpad=+1)
    plt.show()
    plt.close()

