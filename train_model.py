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
from utils.utils import setup

if __name__ == "__main__":
    FLAGS, logger = setup(running_script="./utils/training.py", config="config.json")
    print("FLAGS= ", FLAGS)

    def add_slash(path):
        if path[-1] != "/":
            return path + "/"
        else:
            return path

#    imfolder = add_slash(args.imfolder)
    imfolder = os.path.abspath(FLAGS.input)
    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    print("input dir: ", imfolder,
            "device: : ", device)

    print("Load Data as Tensors...")
    img_dataset = datasets.ImageFolder(
        imfolder,
        transform=transforms.Compose(
            [transforms.ToTensor(),]
            #                transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    data = VAEDataset(img_dataset)

    encoder, decoder = Encoder(z=FLAGS.zdim), Decoder(z=FLAGS.zdim)

    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        network_name=FLAGS.networkname,
        device=device,
        batch_size=FLAGS.batchsize,
        max_epochs=FLAGS.maxpochs,
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
