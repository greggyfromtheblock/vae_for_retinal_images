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
        # Incoming image has shape e.g. 3x192x188
        super(Encoder, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_max_pooling=0):
            return [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=padding_max_pooling)]

        self.conv_layers = nn.Sequential(
            # Formula of new "Image" Size: (origanal_size - kernel_size + 2 * amount_of_padding)//stride + 1
            *conv_block(3, 25, kernel_size=5, stride=1, padding=2),  # (384-5+2*2)//1 + 1 = 384  > Max-Pooling: 384/2=192
            # -> (376-5+2*2)//1 + 1 = 376  --> Max-Pooling: 376/2 = 188
            *conv_block(25, 100, kernel_size=5, stride=1, padding=2),  # (192-5+2*2)//1 + 1 = 192  > Max-Pooling: 190/2=96
            # -> (188-5+2*2)//1 + 1 = 188  --> Max-Pooling: 188/2 = 94
            *conv_block(100, 150, kernel_size=5, padding=2),   # New "Image" Size:  48x47
            *conv_block(150, 200, padding=1),  # New "Image" Size:  24x23
            *conv_block(200, 300, padding=0, padding_max_pooling=1),  # New "Image" Size:  11x10
            *conv_block(300, 400, padding=0, padding_max_pooling=0),  # New "Image" Size:  5x4
            *conv_block(400, 500, padding=0, padding_max_pooling=1),  # New "Image" Size:  2x2
        )

        def linear_block(in_feat, out_feat, normalize=True, dropout=None, negative_slope = None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(nn.BatchNorm1d(out_feat))  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            if negative_slope:
                layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            else:
                layers.append(nn.ReLU())
            return layers

        self.linear_layers = nn.Sequential(
            *linear_block(500 * 2 * 2, 128, normalize=True),
            *linear_block(128, z)
        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        features = features.view(-1, np.prod(features.shape[1:]))
        features = self.linear_layers(features)
        mean = self.mean(features)
        logvar = self.logvar(features)
        return features, mean, logvar

class Decoder(nn.Module):
    def __init__(self, z=32):
        super(Decoder, self).__init__()

        def linear_block(in_feat, out_feat, normalize=True, dropout=None, negative_slope = None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(nn.BatchNorm1d(out_feat))  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            if negative_slope:
                layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            else:
                layers.append(nn.ReLU())
            return layers

        self.linear_blocks1 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, dropout=0.2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks2 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, negative_slope=1e-2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks3 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, dropout=0.2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks4 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, negative_slope=1e-2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks5 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, dropout=0.2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks6 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, negative_slope=1e-2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks7 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, dropout=0.2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks8 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, negative_slope=1e-2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks9 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, dropout=0.2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )
        self.linear_blocks10 = nn.Sequential(
            *linear_block(z, 128, normalize=True),
            *linear_block(128, 256),
            *linear_block(256, 512, negative_slope=1e-2),
            *linear_block(512, 50 * 5 * 4, dropout=0.3),
        )

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, scale_factor=None, size=None, mode='bilinear'):

            layers = [nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2,
                                stride=stride),
                      nn.BatchNorm2d(in_channels),
                      nn.ReLU(),
                      nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                         stride=stride),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU()]

            scale_factor and layers.append(nn.Upsample(mode=mode, scale_factor=scale_factor))
            size and layers.append(nn.Upsample(mode=mode, size=size))
            return layers

        self.conv_layers = nn.Sequential(
            *conv_block(16, 200, padding=1, size=(48, 47)),
            *conv_block(200, 150, padding=1, scale_factor=2, mode='nearest'),
            *conv_block(150, 100, kernel_size=5, padding=2, scale_factor=2, mode='nearest'),
            *conv_block(100, 50, kernel_size=5, padding=2, scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=50, out_channels=3, kernel_size=1),
        )

    def forward(self, latent_vector):
        fc_layers = torch.cat([self.linear_blocks1(latent_vector), self.linear_blocks2(latent_vector), self.linear_blocks3(latent_vector),
                               self.linear_blocks4(latent_vector), self.linear_blocks5(latent_vector), self.linear_blocks6(latent_vector),
                               self.linear_blocks7(latent_vector), self.linear_blocks8(latent_vector), self.linear_blocks9(latent_vector),
                               self.linear_blocks10(latent_vector)], dim=1)
        dec = torch.reshape(fc_layers, (latent_vector.size(0), 16, 25, 25))
        reconstructions = self.conv_layers(dec)
        print(reconstructions.shape)
        return reconstructions


class OdirVAETraining(VAETraining):
    def __init__(self, encoder, decoder, discriminator, data, path_prefix, network_name,
                 # alpha=0.25, beta=0.5, m=120,
                 optimizer=torch.optim.Adam,
                 **kwargs):
        super(OdirVAETraining, self).__init__(
            encoder, decoder, discriminator, data,
            optimizer=optimizer,
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

        if data.size(0) > 24:
            imgs = torch.zeros_like(reconstructions[0:25:5])

            for i in range(0, 5):
                imgs[i] = F.sigmoid(reconstructions[i*5])

            if self.step_id % 20 == 0:
                self.writer.add_images("target", data[0:25:5], self.step_id)
                self.writer.add_images("reconstruction", imgs, self.step_id)
            return mean, logvar, reconstructions, data





if __name__ == '__main__':
    fake_imgs = torch.randn((4, 3, 384, 376))
    encoder = Encoder()
    encoder(fake_imgs)
    fake_latent_vectors = torch.randn((10, 32))
    decoder = Decoder()
    decoder(fake_latent_vectors)
