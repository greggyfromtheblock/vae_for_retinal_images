"""
Trigger training here
"""
import argparse
import os
import sys
from torchvision import datasets, transforms
import numpy as np
import torch
from utils.training import Encoder, Decoder, OdirVAETraining, VAEDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="""Training VAE""")
    parser.add_argument('imfolder', type=str, default=None,
        metavar='image_dir',
                    help="""The path to the directory which contains
                    the imgge folder. The images themselves must be
                    in one or more subdirectories of the imfolder""")
    parser.add_argument('path_prefix', type=str, default=None,
        metavar='path_prefix',
                    help="""The path to the directory which should contain the data for tensorboard.""")
    parser.add_argument('network_name', type=str, default=None,
        metavar='network_name',
                    help="""The name of the network. Use different names for different models!""")
    args = parser.parse_args()
    # python3 train_model.py /home/henrik/PycharmProjects/vae_for_retinal_images/data/processed/train

    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return(path)

    imfolder = add_slash(args.imfolder)
    network_name = args.network_name
    path_prefix = args.path_prefix

    if network_name in os.listdir(path_prefix):
        input1 = input("Network already exists. Are you sure to continue? [y/yes]\n")
        if not input1 in ['y', 'yes']:
            sys.exit()

    print("Load Data as Tensors...")
    img_dataset = datasets.ImageFolder(
        imfolder, transform=transforms.ToTensor()
    )
    data = VAEDataset(img_dataset)

    encoder, decoder = Encoder(), Decoder()
    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        path_prefix=path_prefix,
        net_name=network_name,
        network_name=network_name,
        device = "cuda:3" if torch.cuda.is_available() else "cpu",
        batch_size=50,
        max_epochs=1,
        verbose=True,
    )

    trained = training.train()
    encoder = trained[0]
    sample, _ = img_dataset[0]
    features, mean, logvar = encoder(sample)
    print(type(features))
    if type(features) == torch.Tensor:
        torch.save(features, f"./{network_name}/features.pt")
        torch.save(mean, f"./{network_name}/mean.pt")
        torch.save(logvar, f"./{network_name}/logvar.pt")

    if (type(features)) == np.ndarray:
        np.save(f"./{network_name}/features.npy", features )
        np.save(f"./{network_name}/mean.npy", mean)
        np.save(f"./{network_name}/logvar.npy", logvar)

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
