"""
Trigger training here
"""
import os
import sys
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from skimage import io
from tqdm import tqdm
from utils.training import Encoder, Decoder, OdirVAETraining, VAEDataset
from utils.utils import setup

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


if __name__ == "__main__":
    FLAGS, logger = setup(running_script="./utils/training.py", config="config.json")
    print("FLAGS= ", FLAGS)

    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return(path)

#    imfolder = add_slash(args.imfolder)
    imfolder = os.path.abspath(FLAGS.input)
    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    if network_name in os.listdir(path_prefix):
        input1 = input("Network already exists. Are you sure to continue? [y/yes]\n")
        if not input1 in ['y', 'yes']:
            sys.exit()

    print("input dir: ", imfolder,
            "device: : ", device)

    print("Load Data as Tensors...")
    transform_data = transforms.Compose([transforms.ToTensor(), normalize])

    img_dataset = datasets.ImageFolder(
        imfolder, transform=transform_data
    )

    data = VAEDataset(img_dataset)

    encoder, decoder = Encoder(z=FLAGS.zdim), Decoder(z=FLAGS.zdim)


    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        network_name=FLAGS.networkname,
        path_prefix=FLAGS.path_prefix,
        device=device,
        batch_size=FLAGS.batchsize,
        max_epochs=FLAGS.maxpochs,
        verbose=True,
    )

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
