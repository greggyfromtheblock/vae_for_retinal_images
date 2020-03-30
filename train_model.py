"""
Trigger training here
"""
import argparse
import os
import sys
from torchvision import datasets, transforms
import numpy as np
import torch
from skimage import img_as_ubyte
from utils.training import Encoder, Decoder, OdirVAETraining, VAEDataset


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="""Training VAE""")
    parser.add_argument('imfolder', type=str, default=None,
        metavar='image_dir',
                    help="""The path to the directory which contains
                    the imgge folder. The images themselves must be
                    in one or more subdirectories of the imfolder""")
    parser.add_argument('--path_prefix', type=str, default=".",
        metavar='path_prefix',
                    help="""The path to the directory which should contain the data for tensorboard.""")
    parser.add_argument('network_name', type=str, default=None,
        metavar='network_name',
                    help="""The name of the network. Use different names for different models!""")
    parser.add_argument('--gpu_number', type=str, default=None,
        metavar='network_name',
                    help="""The gpu you want to use. Number must be between 0 and 7""")
    args = parser.parse_args()

    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return(path)

    imfolder = add_slash(args.imfolder)
    network_name = args.network_name
    path_prefix = args.path_prefix  # optional argument. If default: the path is the current one.
    gpu_number = int(args.gpu_number) if int(args.gpu_number) in range(8) else 3

    if network_name in os.listdir(path_prefix):
        input1 = input("Network already exists. Are you sure to continue? [y/yes]\n")
        if not input1 in ['y', 'yes']:
            sys.exit()

    print("Load Data as Tensors...")
    transform_data = transforms.Compose([transforms.ToTensor(), normalize])
    img_dataset = datasets.ImageFolder(
        imfolder, transform=transform_data
    )

    data = VAEDataset(img_dataset)
    encoder, decoder = Encoder(), Decoder()
    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        path_prefix=path_prefix,
        network_name=network_name,
        device="cuda:%i" % gpu_number if torch.cuda.is_available() else "cpu",
        batch_size=128,
        max_epochs=100,
        verbose=True
    )
    print(len(data), data[0][0].shape)
    print("To check if values are between 0 and 1:\n", data[0][0][0][50][30:180:10])

    print("Start Training...")
    trained = training.train()
    print("Finished Training...")

    encoder = trained[0]
    sample, _ = img_dataset[0]
    features, mean, logvar = encoder(sample)

    print(type(features))
    if type(features) == torch.Tensor:
        torch.save(features, f"{path_prefix}/{network_name}/features.pt")
        torch.save(mean, f"{path_prefix}/{network_name}/mean.pt")
        torch.save(logvar, f"{path_prefix}/{network_name}/logvar.pt")

    if (type(features)) == np.ndarray:
        np.save(f"{path_prefix}/{network_name}/features.npy", features)
        np.save(f"{path_prefix}/{network_name}/mean.npy", mean)
        np.save(f"{path_prefix}/{network_name}/logvar.npy", logvar)


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
