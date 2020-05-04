"""
Add the Training (TorchSupport-Training API) and loss functions here.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torchsupport.training.vae import VAETraining
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from torchsupport.training.training import SupervisedTraining

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# for Resnet
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

from torchsummary import summary

import torchvision.models as models  # ppretrained resnet etc.
from torchvision import datasets, transforms as T

from collections import OrderedDict

import torchsupport.modules.losses.vae as vl


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


class Encoder(nn.Module):
    def __init__(self, z=32, pretrained=False):
        super(Encoder, self).__init__()
        self.z = z

        # model = resnet101(3, z).cuda() #3: rgb input channels, 32: latent space dim
        # model = resnetCustom(3,z).cuda()
        model = models.resnet101(pretrained=pretrained).cuda()  # output is [-1, 1000]

        # model = models.wide_resnet101_2(pretrained=True).cuda() #output is [-1, 1000]
        # freeze the weights because we are using pretrained model:
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False
        # change last layer to fit zdim (by default it will be requires_grad=T
        # model.fc = nn.Linear(model.fc.in_features, z, bias=True)

        self.encoder = model

    def forward(self, inputs):
        # inputs = inputs.view(inputs.size(0), -1)
        features = self.encoder(inputs)
        return features


class OdirSuperVisedTraining(SupervisedTraining):
    def __init__(
        self,
        net,
        train_data,
        validate_data,
        losses,
        optimizer=torch.optim.Adam,
        schedule=None,
        max_epochs=50,
        batch_size=128,
        accumulate=None,
        device="cpu",
        network_name="network",
        path_prefix=".",
        report_interval=10,
        checkpoint_interval=1000,
        valid_callback=lambda x: None,
        criterion=nn.BCEWithLogitsLoss(reduction="sum"),
        **kwargs,
    ):
        super(OdirSuperVisedTraining, self).__init__(
            optimizer,
            schedule,
            max_epoch,
            batch_size,
            accumulate,
            device,
            network_name,
            path_prefix,
            report_interval,
            checkpoint_interval,
            valid_callback,
            **kwargs,
        )
        self.checkpoint_path = f"{path_prefix}/{network_name}/{network_name}-checkpoint"
        self.writer = SummaryWriter(f"{path_prefix}/{network_name}/")
        self.epoch = None
        self.criterion = criterion

    def myloss(self, data):
        pass

    def run_networks(self, data, *arg, **kwargs):
        inputs, labels = data
        predictions = self.net(inputs)
        return [combined for combined in zip(predictions, labels)]

    def loss(self, inputs):
        """inputs is a list of length batch_size,
        each element is a tuple (predicts,labels)"""
        loss_val = torch.tensor(0.0).to(self.device)
        for pred, lab in inputs:
            loss_val += self.criterion(pred, lab)
        return loss_val
