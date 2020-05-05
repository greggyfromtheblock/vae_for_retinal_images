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

# from utils.training_supervised import (
#    RetinnSuperVisedDataset,
#    SupervisedTraining,
# )


################### definitions #########################
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


def add_slash(path):
    if path[-1] != "/":
        return path + "/"
    else:
        return path


def bce(predict, target):
    result = F.binary_cross_entropy_with_logits(predict, target, reduction="sum")
    return result


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
        self.csv_file = csv_file
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
        image = io.imread(os.path.abspath(self.imdir) + "/" + img_name)
        if self.transform != None:
            image = self.transform(image)
        cols = self.labels.columns[5:13]  # N,D,...,O
        labels = self.labels.loc[img_name, cols]
        labels = torch.Tensor(labels)
        sample = {"image": image, "labels": labels, "name": img_name}
        return sample["image"], sample["labels"]


class SupervisedCustomTraining:
    """Give it a dataset, parameters and shit"""

    def __init__(
        self,
        dataset,
        net,
        network_name="myNetwork",
        figures_dir="myNetworkFiguresAreAwesomeMehtaWorldPeace",
        optimizer=optim.Adam,
        maxpoch=10,
        device="cpu",
        batch_size=64,
        learning_rate=5e-5,
        criterion=nn.BCEWithLogitsLoss(),
    ):
        self.dataset = dataset
        self.figures_dir = figures_dir
        self.batch_size = batch_size
        self.network_name = network_name
        self.figures_dir = figures_dir
        self.maxpoch = maxpoch
        self.learning_rate = (learning_rate,)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = net.to(device=self.device)
        self.criterion = criterion.to(device=self.device)
        self.optimizer = optimizer(self.net.parameters(),)
        print("loaded model and parameters")
        print(model)


def train_model(
    model,
    dataloaders,
    optimizer,
    num_epochs=25,
    criterion=nn.BCEWithLogitsLoss(reduction="sum"),
    is_inception=False,
):
    """
           The train_model function handles the training and validation of a given
        model. As input, it takes a PyTorch model, a dictionary of dataloaders, a loss
        function, an optimizer, a specified number of epochs to train and validate for,
        and a boolean flag for when the model is an Inception model. The is_inception
        flag is used to accomodate the Inception v3 model, as that architecture uses an
        auxiliary output and the overall model loss respects both the auxiliary output
        and the final output, as described here. The function trains for the specified
        number of epochs and after each epoch runs a full validation step. It also keeps
        track of the best performing model (in terms of validation accuracy), and at the
        end of training returns the best performing model. After each epoch, the
        training and validation accuracies are printed.
    """
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == "train":
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        print("shape of outputs", outputs.shape)
                        loss = criterion(outputs, labels)

                    #_, preds = torch.max(outputs, 1)
                    preds = torch.max(outputs, torch.ones(8).to(device)).to(device)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                print("shpe of preds: ", preds.shape, "shape of labels: ",
                        labels.shape)
                running_corrects += torch.sum(preds == labels.data)
                #running_corrects += torch.sum(preds == labels)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


### Tests #####

csv_file = "../retina/outputs/supervised_sets/odir-training.csv"
test_dir = "../retina/outputs/supervised_sets/training_images/images/"
valid_dir= "../retina/outputs/supervised_sets/validation_images/images/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
zdim = 8
num_epochs=2

plt.ion()

#df = pd.read_csv(csv_file, header=0, index_col="Fundus Image", sep="\t")
#df[df["anterior"] == 1]
#df[df["no fundus"] == 1]

plt.show()

input_size=224

#mytransform = T.Compose([T.CenterCrop(input_size), T.ToTensor(), normalize])
#mytransform = T.Compose([T.CenterCrop(input_size)])

mytransform = T.Compose([T.ToPILImage(),
    T.CenterCrop(input_size),
    T.ToTensor(), normalize])

#prepare model
model = models.resnet101(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, zdim, bias=True)

#test and validation datasets
test_dataset = RetinnSuperVisedDataset(test_dir, csv_file, transform=mytransform)
valid_dataset = RetinnSuperVisedDataset(valid_dir, csv_file, transform=mytransform)

xx,yy = test_dataset.__getitem__(1)

xx
yy

#put them in a dictionary:
image_datasets = {'train' : test_dataset, 'val' : valid_dataset}

dataloaders_dict = {x : DataLoader(image_datasets[x],
    batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train',
    'val']}

model.to(device)

feature_extract=False

params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update)

criterion=nn.BCEWithLogitsLoss(reduction='sum')

model, hist = train_model(model, dataloaders_dict, optimizer_ft,
        num_epochs=num_epochs, is_inception=False)




test_train = SupervisedCustomTraining(test_dataset, model)
test_dataloader = DataLoader(dataset=test_train, batch_size=64)

valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64)

train_model(
    model=model,
    dataloaders={"train": test_dataloader, "valid": valid_dataloader},
)

# dataloader = DataLoader(dataset=data, batch_size=5)
# dataiter = iter(dataloader)
# test_datatset = datasets.MNIST(
#    root=".testmnist/", download=True, transform=T.ToTensor()
# )
# valid_dataset = datasets.MNIST(
#    root=".testmnistvalid/", train=False, transform=T.ToTensor(), download=True
# )
# test_dataloader = DataLoader(dataset=test_datatset, batch_size=5)
# valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=5)
# testdataiter = iter(test_dataloader)
# model = models.resnet101(pretrained=False)

# test_training = SupervisedTraining(model, test_datatset, valid_dataset, losses=[bce])
# test_training = SupervisedTraining(
#    model, test_dataloader, valid_dataloader, losses=[bce]
# )

#################
