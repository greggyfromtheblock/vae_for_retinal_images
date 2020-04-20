"""
Add the Training (TorchSupport-Training API) and loss functions here.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsupport.training.vae import VAETraining
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class VAEDataset(Dataset):
    def __init__(self, data):
        super(VAEDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        data, label = self.data[index]
        return data,

    def __len__(self):
        return len(self.data)
        pass


class Encoder(nn.Module):
    def __init__(self, z=32):
        # Incoming image has shape e.g. 384x376x3
        super(Encoder, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_max_pooling=0):
            return [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=padding_max_pooling)]

        self.conv_layers = nn.Sequential(
            # Formula of new "Image" Size: (origanal_size - kernel_size + 2 * amount_of_padding)//stride + 1
            *conv_block(3, 12, kernel_size=5, stride=1, padding=2), # (384-5+2*2)//1 + 1 = 384  > Max-Pooling: 384/2=192
            # -> (376-5+2*2)//1 + 1 = 376  --> Max-Pooling: 376/2 = 188
            *conv_block(12, 24, kernel_size=5, stride=1, padding=2), # (192-5+2*2)//1 + 1 = 192  > Max-Pooling: 190/2=96
            # -> (188-5+2*2)//1 + 1 = 188  --> Max-Pooling: 188/2 = 94
            *conv_block(24, 48, kernel_size=5, padding=2),   # New "Image" Size:  48x47
            *conv_block(48, 96, padding=1),  # New "Image" Size:  24x23
            *conv_block(96, 128, padding=0, padding_max_pooling=1),  # New "Image" Size:  11x10
            *conv_block(128, 256, padding=0, padding_max_pooling=0),  # New "Image" Size:  5x4
            *conv_block(256, 512, padding=0, padding_max_pooling=1),  # New "Image" Size:  2x2
        )

        def linear_block(in_feat, out_feat, normalize=True, dropout=None, negative_slope=1e-2):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(nn.BatchNorm1d(out_feat))  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            return layers

        self.linear_layers = nn.Sequential(
            *linear_block(512 * 2 * 2, 256, normalize=True, dropout=0),
            *linear_block(256, 128, dropout=0),
            *linear_block(128, 96),
            *linear_block(96, 96),
            *linear_block(96, 48),
            *linear_block(48, z, negative_slope=0.0)
        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        # print(features.shape)
        features = features.view(-1, np.prod(features.shape[1:]))
        # print(features.shape)
        features = self.linear_layers(features)
        # print(8,features.shape)
        mean = self.mean(features)
        logvar = self.logvar(features)
        return features, mean, logvar

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
            *linear_block(z, 48, normalize=True),
            *linear_block(48, 96),
            *linear_block(96, 128),
            *linear_block(128, 256, dropout=None),
            *linear_block(256, 512, dropout=0),
            *linear_block(512, 512 * 2 * 2, dropout=0),
            nn.ReLU()
        )

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, scale_factor=None, size=None, mode='bilinear'):

            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                         stride=stride),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU()]

            scale_factor and layers.append(nn.Upsample(mode=mode, scale_factor=scale_factor))
            size and layers.append(nn.Upsample(mode=mode, size=size))
            return layers

        self.conv_layers = nn.Sequential(
            *conv_block(512, 256, padding=1, size=(5, 4)),
            *conv_block(256, 128, padding=1, size=(11, 10)),
            *conv_block(128, 96, padding=1, size=(24, 23), mode='nearest'),
            *conv_block(96, 64, padding=1, size=(48, 47), mode='nearest'),
            *conv_block(64, 32, padding=1, scale_factor=2),
            *conv_block(32, 12, kernel_size=5, padding=2, scale_factor=2, mode='nearest'),
            *conv_block(12, 6, kernel_size=5, padding=2, scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1),
        )

    def forward(self, latent_vector):
        dec = torch.reshape(self.linear_blocks(latent_vector), (latent_vector.shape[0], 512, 2, 2))
        reconstructions = self.conv_layers(dec)
        # print(reconstructions.shape)
        return reconstructions


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


class OdirVAETraining(VAETraining):
    def __init__(self, encoder, decoder, data, path_prefix, network_name,
                 # alpha=0.25, beta=0.5, m=120,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={"lr": 5e-5},
                 **kwargs):
        super(OdirVAETraining, self).__init__(
            encoder, decoder, data,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs
        )
        self.checkpoint_path = f"{path_prefix}/{network_name}/{network_name}-checkpoint"
        self.writer = SummaryWriter(f"{path_prefix}/{network_name}/")
        self.epoch = None

    def run_networks(self, data, *args):
        mean, logvar, reconstructions, data = super().run_networks(data, *args)

        if self.epoch != self.epoch_id:
            self.epoch = self.epoch_id
            print("%i-Epoch" % (self.epoch_id+1))

        imgs = torch.zeros_like(reconstructions[0:50:10])

        for i in range(0, 5):
            imgs[i] = F.sigmoid(reconstructions[i*10])

        if self.step_id % 20 == 0:
            self.writer.add_images("target", data[0:50:10], self.step_id)
            self.writer.add_images("reconstruction", imgs, self.step_id)
        return mean, logvar, reconstructions, data


if __name__ == '__main__':
    fake_imgs = torch.randn((4, 3, 384, 376))
    encoder = Encoder()
    encoder(fake_imgs)
    fake_latent_vectors = torch.randn((10, 32))
    decoder = Decoder()
    decoder(fake_latent_vectors)
