"""
Trigger training here
"""
import argparse
import os
import sys
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from skimage import io
from tqdm import tqdm
import torch
from utils.training import Encoder, Decoder, OdirVAETraining, VAEDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Training VAE""")
    parser.add_argument("--usecuda", action="store_true", help="""try to use cuda""")
    parser.add_argument(
        "imfolder",
        type=str,
        default=None,
        metavar="image_dir",
        help="""The path to the directory which contains
                    the imgage folder. The images themselves must be
                    in one or more subdirectories of the imfolder""",
    )
    args = parser.parse_args()

    def add_slash(path):
        if path[-1] != "/":
            return path + "/"
        else:
            return path

    imfolder = add_slash(args.imfolder)

    print("Load Data as Tensors...")
    #    img_dataset = datasets.ImageFolder(
    #        "./data/processed/", transform=transforms.ToTensor()
    #    )
    img_dataset = datasets.ImageFolder(
        imfolder,
        transform=transforms.Compose(
            [transforms.ToTensor(),]
            #                transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    data = VAEDataset(img_dataset)

    encoder, decoder = Encoder(z=32), Decoder(z=32)

    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        network_name="odir-vae",
        device="cuda" if (args.usecuda and torch.cuda.is_available()) else "cpu",
        batch_size=64,
        max_epochs=700,
        verbose=True,
    )

    training.train()


"""    
def prepare_datasets(logger, path_to_splits):
datasets = {'train': ''}
return datasets

FLAGS, logger = setup(running_script="train_ECG_vae.py",
                      config='config.json')

# input
split_data_path = FLAGS.input.strip().split(',')

datasets, eids = prepare_datasets(logger, split_data_path)

trained = train(logger, FLAGS, datasets['train'])

logger.info('Done.')
"""
