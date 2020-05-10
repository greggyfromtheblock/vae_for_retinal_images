# Standard Python Modules
from __future__ import print_function, division
import os
import sys
import time
import argparse
import copy

# pndas, plt, numpy, and other
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Standard Torch Modules
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# torchvision
import torchvision
from torchvision import datasets, transforms as T
from torchvision import utils as vutils
from torchvision.utils import save_image
from torchvision import models

# Torchsupport
import torchsupport
from torchsupport.training.vae import VAETraining
from torchsupport.training.training import SupervisedTraining

# Augmentor
import Augmentor

# skimage.sklearn
from skimage import io, transform as skT

# other
from tensorboardX import SummaryWriter
from torchsummary import summary

# for Resnet
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

# Own Modules
from utils.utils import setup
from utils.get_mean_std import get_mean_std

"""
Trigger training here
"""
import argparse
import os
import sys
import time
from torchvision import datasets, transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from skimage import io
from tqdm import tqdm
import torch

# from utils.training import Encoder, Decoder, OdirVAETraining, VAEDataset
from utils.training_resnet_pretrainedv02 import (
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


################################## Tests

import json
config="config.v02.json"

from utils.utils import set_default_options

with open(config) as f:
  jdata = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(jdata)

jdata.keys()
jdata.values()

plt.ion()

imfolder = 'smalldata/250x320/'
network_name = 'test_betwork' 
path_prefix = 'test_prefix'
network_dir = f"{path_prefix}/{network_name}/"
device = FLAGS.device if torch.cuda.is_available() else "cpu"

myTransform1 = T.Compose(
    [
        #T.Grayscale(3),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
        #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #T.Normalize(means, stds),
    ]
    #                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    #                T.Normalize((0.5,), (0.5,))]
)


myTransform2 = T.Compose([
#    T.ToPILImage(),
    T.CenterCrop(224), #for resnet
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


myTransform3 = T.Compose([
#    T.ToPILImage(),
    T.CenterCrop(224), #for resnet
    T.ToTensor(),
    normalize,
    ])

myTransform4 = T.Compose([
    T.CenterCrop(224), #for resnet
    T.ToTensor(),
    ])


img_dataset = datasets.ImageFolder(
    imfolder,
    transform=myTransform1,
)

data = VAEDataset(img_dataset)



plt.show()

img2 = io.imread('smalldata/250x320/images/14_left.jpg')
io.imshow(img2)
#bimgs = torch.ones((2,3,224,224))

img, _ = img_dataset.__getitem__(33)
y = img.numpy()
y = y.transpose((1,2,0))
io.imshow(y)



y
################################## 

if __name__ == "__main__":

    FLAGS, logger = setup(running_script="./utils/training.py",
            config="config.v02.json")
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
    f = T.Normalize(means, stds)
    f_inv = T.Normalize(mean = -means/stds, std = 1/stds)


    myOldtransform=T.Compose(
        [
            #T.Grayscale(3),
            #T.CenterCrop(224),
            T.ToTensor(),
            normalize,
            #T.Normalize(means, stds),
        ]
        #                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        #                T.Normalize((0.5,), (0.5,))]
    )


#    mytransform = T.Compose([
#    #    T.ToPILImage(),
#        T.CenterCrop(224), #for resnet
#        T.ToTensor(),
#        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#        ])


    mytransform = T.Compose([
    #    T.ToPILImage(),
        T.CenterCrop(224), #for resnet
        T.ToTensor(),
        normalize,
        ])


    img_dataset = datasets.ImageFolder(
        imfolder,
        transform=mytransform,
    )
    data = VAEDataset(img_dataset)

    imsize = tuple([int(i) for i in FLAGS.image_dim.split(',')])
    print("using state_dict from:", FLAGS.state_dict)
    encoder = Encoder(z=FLAGS.zdim, pretrained=FLAGS.pretrained,
            state_dict=FLAGS.state_dict)
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
        #in_trans = f_inv,
        #out_trans = T.Compose(
        #    [torch.nn.functional.sigmoid, ])
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
