# Standard Python Modules
from __future__ import print_function, division
import os
import sys
import time
import argparse

# pndas, plt, numpy, and other 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Standard Torch Modules
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

#torchvision
import torchvision
from torchvision import datasets, transforms as T
from torchvision import utils as vutils
from torchvision.utils import save_image

# Torchsupport
import torchsupport
from torchsupport.training.vae import VAETraining
from torchsupport.training.training import SupervisedTraining
# Augmentor
import Augmentor
#skimage.sklearn
from skimage import io, transform as skT
#other
from tensorboardX import SummaryWriter
from torchsummary import summary
# for Resnet
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

# Own Modules
from utils.utils import setup
from utils.get_mean_std import get_mean_std

#from utils.training_supervised import (
#    RetinnSuperVisedDataset,
#    SupervisedTraining,
#)

### Tests #####
csv_file = "data/odir_training_annotations.csv"

imdir = "smalldata/smalloutputdir2/"

df = pd.read_csv(
    "data/odir_training_annotations.csv", header=0, index_col="Fundus Image", sep="\t"
)
names = os.listdir(imdir)

onames = [reconstructFileName(f) for f in names]

df.loc[onames]

df2 = df.loc[onames]
df2.index = names

df3 = addAugmentationAnnotations(imdir, csv_file)

plt.ion()

img = io.imread(imdir + "7_left.jpg")

io.imshow(img)

plt.imshow(img)

plt.show()

cols = df3.columns

mytransform = T.Compose([T.ToTensor(), normalize])

data = RetinnSuperVisedDataset(imdir, csv_file)

data.__getitem__(1)


dataloader = DataLoader(dataset=data, batch_size=5)

dataiter = iter(dataloader)

test_datatset = datasets.MNIST(
    root=".testmnist/", download=True, transform=T.ToTensor()
)

test_dataloader = DataLoader(dataset=test_datatset, batch_size=5)

testdataiter = iter(test_dataloader)
#################


def reconstructFileName(s):
    """give it something like '9_left_rot_foo.jpg'
    and it returns '9_left.jpg', that was the original
    file name before augmentation."""
    s = s.replace(".jpg", "")
    s = s.split("_")
    s = s[:2]
    s = "_".join(s)
    s += ".jpg"
    return s


def addAugmentationAnnotations(imdir, csv_file):
    """Give it the path for the image dir and a the 
    path for the annotations. 
    The image dir contains all the original images plus the 
    augmentations, whcih by convention are names
    'original_name_augmentationsuffix.jpg'
    The function adds entries for the augmentations which are
    duplications of the original image labels (except their name/imdex)"""
    df = pd.read_csv(csv_file, header=0, sep="\t", index_col="Fundus Image")
    # df = df.loc[df.index.intersection(names)]
    names = os.listdir(imdir)
    onames = [reconstructFileName(f) for f in names]
    df2 = df.loc[onames]
    df2.index = names
    return df2


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


class RetinnSuperVisedDataset(Dataset):
    """Expects to get a path for a directory containing images,
    and path for a csv file with annotations. Every image file 
    should have annotations but there may be more annotations then
    images in the csv. Optionally also provide transform to perform
    on the images"""

    def __init__(self, imdir, csv_file, transform=T.ToTensor()):
        # df = pd.read_csv(csv_file, header=0,
        #        index_col='Fundus Image',
        #        sep='\t' )
        self.imdir = imdir
        self.transform = transform
        self.imnames = os.listdir(self.imdir)
        self.labels = addAugmentationAnnotations(imdir, csv_file)

    def __len__(self):
        l = os.listdir(self.imdir)
        return len(l)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.imnames[idx]
        image = io.imread(os.path.abspath(imdir) + "/" + img_name)
        if self.transform != None:
            image = self.transform(image)
        cols = self.labels.columns[5:13]  # N,D,...,O
        labels = self.labels.loc[img_name, cols]
        labels = torch.Tensor(labels)
        sample = {"image": image, "labels": labels, "name": img_name}
        return sample["image"], sample["labels"]

class RetinnSupervisedTraining:
    pass
