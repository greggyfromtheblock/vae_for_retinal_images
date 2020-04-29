"""
Trigger training here
"""
import argparse
import os
import sys
import time
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from skimage import io
from tqdm import tqdm
import torch

# from utils.training import Encoder, Decoder, OdirVAETraining, VAEDataset
from utils.training_resnet_pretrained import (
    Encoder,
    Decoder,
    OdirVAETraining,
    VAEDataset,
)
from utils.utils import setup

from utils.get_mean_std import get_mean_std


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def add_slash(path):
    if path[-1] != "/":
        return path + "/"
    else:
        return path


if __name__ == "__main__":
    FLAGS, logger = setup(running_script="./utils/training.py", config="config.json")
    print("FLAGS= ", FLAGS)

    #    imfolder = add_slash(args.imfolder)
    imfolder = os.path.abspath(FLAGS.input)
    network_name = FLAGS.network_name
    path_prefix = FLAGS.path_prefix
    network_dir = f"{path_prefix}/{network_name}/"
    device = FLAGS.device if torch.cuda.is_available() else "cpu"

    print("input dir: ", imfolder, "device: : ", device)

    # os.makedirs(FLAGS.path_prefix, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)
    # if FLAGS.networkname in os.listdir(FLAGS.path_prefix):
    if FLAGS.network_name in os.listdir(network_dir):
        input1 = input("\nNetwork already exists. Are you sure to proceed? ([y]/n) ")
        if not input1 in ["y", "yes"]:
            sys.exit()

    print("Load Data as Tensors...")
    means, stds = get_mean_std(imfolder)
    f = transforms.Normalize(means, stds)
    f_inv = transforms.Normalize(mean = -means/stds, std = 1/stds)
    img_dataset = datasets.ImageFolder(
        imfolder,
        transform=transforms.Compose(
            [
                #transforms.Grayscale(3),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ]
            #                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            #                transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    data = VAEDataset(img_dataset)

    imsize = tuple([int(i) for i in FLAGS.image_dim.split(',')])
    encoder = Encoder(z=FLAGS.zdim, pretrained=FLAGS.pretrained)
    decoder = Decoder(z=FLAGS.zdim, imsize=imsize)

    training = OdirVAETraining(
        encoder,
        decoder,
        data,
        path_prefix=path_prefix,
        network_name=network_name,
        device=device,
        optimizer_kwargs={"lr": FLAGS.learningrate},
        batch_size=FLAGS.batchsize,
        max_epochs=FLAGS.maxpochs,
        verbose=True,
        in_trans = f_inv,
        out_trans = transforms.Compose(
            [torch.nn.functional.sigmoid, ])
    )

    print(
        "\nSize of the dataset: {}\nShape of the single tensors: {}".format(
            len(data), data[0][0].shape
        )
    )
    #    print(
    #        "\nTo check if values are between 0 and 1:\n{}".format(
    #            data[0][0][0][50][30:180:10]
    #        )
    #    )

    print("\nStart Training...")
    time_start = time.time()
    trained_encoder, _ = training.train()
    print(
        "\nTraining with %i epochs done! Time elapsed: %.2f minutes"
        % (FLAGS.maxpochs, (time.time() - time_start) / 60)
    )

    # print(trained_encoder)

    # TODO: Also refactor path_prefix/networkname into args/FLAGS
    # Save network
    # PATH = f"{FLAGS.path_prefix}/{FLAGS.networkname}/{FLAGS.networkname}.pth"
    PATH = network_dir + f"{network_name}.pth"
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(trained_encoder.state_dict(), PATH)

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
