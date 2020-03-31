"""
Add the Training (TorchSupport-Training API) and loss functions here.
"""
from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, BatchSampler
from torchvision.transforms import ToTensor, Resize
from torchsupport.training.vae import VAETraining

import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


class VAEDataset(Dataset):
  def __init__(self, data):
    self.data = data
  def __getitem__(self, index):
    data, label = self.data[index]
    return (data,)
  def __len__(self):
    return len(self.data)
    pass


class Encoder(nn.Module):
    def __init__(self, z=32):
        # Incoming image has shape e.g. 920x920x3  (Assumption: quadratic image)
        super(Encoder, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
            return [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.MaxPool2d(kernel_size=2, stride=2)]

        self.conv_layers = nn.Sequential(
            # Formula of new Image_Size: (origanal_size - kernel_size + 2 * amount_of_padding)//stride + 1
            *conv_block(3, 8, kernel_size=5, stride=1),   # -> (920-5+0)//2 + 1 = 916  --> Max-Pooling: 916/2 = 458
            *conv_block(8, 12, kernel_size=5, padding=1),   # New Image_Size:  (458-5+2*1)/1 + 1 = 456 --> 456/2 = 228
            *conv_block(12, 16, padding=1),   # New Image_Size:  (228-3+2*1)/1 + 1 = 228 --> Max-Pooling: 228/2 = 114
            *conv_block(16, 24),   # New Image_Size:  (114-3+2*0)/1 + 1 = 112 --> Max-Pooling: 112/2 = 56
            *conv_block(24, 32, padding=1),   # New Image_Size:  (56-3+2*1)/1 + 1 = 56 --> Max-Pooling: 56/2 = 28
            *conv_block(32, 38, padding=1),   # New Image_Size:  (28-3+2*1)/1 + 1 = 28 --> Max-Pooling: 28/2 = 14
            *conv_block(38, 44),   # New Image_Size:  (14-3+2*0)/1 + 1 = 12 --> Max-Pooling: 26/2 = 6
        )

        def linear_block(in_feat, out_feat, normalize=True, dropout=None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(nn.BatchNorm1d(out_feat))  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.linear_layers = nn.Sequential(
            # output_channels: 44; 6 x 6 from image dimension; 44*6*6 = 1584
            *linear_block(44 * 6 * 6, 512, normalize=False, dropout=0.5),
            *linear_block(512, 256, dropout=0.5),
            *linear_block(256, 128),
            *linear_block(128, 64),
            nn.Linear(64, z),
            nn.ReLU()
        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        print(features.shape)
        features = features.view(-1, self.num_flat_features(features))
        print(features.shape)
        features = self.linear_layers(features)

        mean = self.mean(features)
        logvar = self.logvar(features)
        print(features.shape)
        return features, mean, logvar

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Decoder(nn.Module):
    def __init__(self, z=32):
        super(Decoder, self).__init__()

        def linear_block(in_feat, out_feat, normalize=True, dropout=None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(nn.BatchNorm1d(out_feat))  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.linear_blocks = nn.Sequential(
            *linear_block(z, 64, normalize=False),
            *linear_block(64, 256),
            *linear_block(256, 512, dropout=0.5),
            *linear_block(512, 1584, dropout=0.5),   # 44*6*6 = 1584
            nn.ReLU()
        )

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
            return [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Upsample(mode='bilinear', scale_factor=2)]

        self.conv_layers = nn.Sequential(
            *conv_block(44, 36, padding=1),
            *conv_block(36, 30, padding=1),
            *conv_block(30, 24,  kernel_size=5, padding=1),
            *conv_block(24, 20, kernel_size=5, padding=1),
            *conv_block(20, 12, kernel_size=5),
            *conv_block(12, 8, kernel_size=5),
            *conv_block(8, 3, kernel_size=5),
            # nn.UpsamplingNearest2d(size=(920, 920)),  # The wished size
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
        )

    def forward(self, latent_vector):
        dec = torch.reshape(self.linear_blocks(latent_vector), (latent_vector.shape[0], 44, 6, 6))
        print(dec.shape)
        reconstructions = self.conv_layers(dec)
        print(reconstructions.shape)
        return reconstructions


class OdirVAETraining(VAETraining):
    def run_networks(self, data, *args):
        mean, logvar, reconstructions, data = super().run_networks(data, *args)
        self.writer.add_image("target", normalize(data[0]), self.step_id)
        self.writer.add_image("reconstruction", normalize(reconstructions[0].sigmoid()), self.step_id)
        return mean, logvar, reconstructions, data


if __name__ == "__main__":
    # Test Encoder
    fake_imgs = torch.randn((10, 3, 920, 920))
    print(fake_imgs.shape)
    encoder = Encoder()
    encoder.forward(fake_imgs)

    # Test Decoder
    fake_latent_vector = torch.randn((10, 32))
    print(fake_latent_vector.shape)
    decoder = Decoder()
    decoder.forward(fake_latent_vector)

    # Test with fake data:
    batch_size = 10
    fake_imgs = torch.randn((batch_size, 3, 920, 920))


    img_dataset = datasets.ImageFolder('./train_set2/',
            transform=transforms.ToTensor())

    data = VAEDataset(img_dataset)
    #data = TensorDataset(fake_imgs)

    training = OdirVAETraining(
        encoder, decoder, data,
        network_name="odir-vae",
        device="cpu",
        batch_size=3,
        max_epochs=2,
        verbose=True
    )

    training.train()
