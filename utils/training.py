"""
Add the Training (TorchSupport-Training API) and loss functions here.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsupport.training.vae import VAETraining
import torch.nn.functional as F
import torchsupport.modules.losses.vae as vl
from torchsupport.data.io import netwrite, to_device

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
        # Incoming image has shape e.g. 192x188x3
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
            *conv_block(3, 32, kernel_size=5, stride=1, padding=2),  # (192-5+2*2)//1 + 1 = 192  > Max-Pooling: 190/2=96
            # -> (188-5+2*2)//1 + 1 = 188  --> Max-Pooling: 188/2 = 94
            *conv_block(64, 64, kernel_size=5, padding=2),   # New "Image" Size:  48x47
            *conv_block(64, 96, padding=1),  # New "Image" Size:  24x23
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
            *linear_block(512 * 2 * 2, 512, normalize=True, dropout=0.6),
            *linear_block(512, 512, normalize=True, dropout=0.5),
            *linear_block(512, 256, dropout=0.4),
            *linear_block(256, 128),
            *linear_block(128, 128),
            *linear_block(128, 64),
            *linear_block(64, z, negative_slope=0.0)
        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        # print(features.shape)
        # features = features.view(-1, self.num_flat_features(features))
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
            *linear_block(z, 64, normalize=True),
            *linear_block(64, 128),
            *linear_block(128, 128),
            *linear_block(128, 256, dropout=None),
            *linear_block(256, 256, dropout=0.3),
            *linear_block(256, 512, dropout=0.3),
            *linear_block(512, 512 * 2 * 2, dropout=0.4),
            nn.ReLU()
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
            *conv_block(512, 256, padding=1, size=(5, 4)),
            *conv_block(256, 128, padding=1, size=(11, 10)),
            *conv_block(128, 96, padding=1, size=(24, 23), mode='nearest'),
            *conv_block(96, 64, padding=1, size=(48, 47), mode='nearest'),
            *conv_block(64, 32, padding=1, scale_factor=2),
            *conv_block(32, 8, kernel_size=5, padding=2, scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1),
        )

    def forward(self, latent_vector):
        dec = torch.reshape(self.linear_blocks(latent_vector), (latent_vector.shape[0], 512, 2, 2))
        reconstructions = self.conv_layers(dec)
        return reconstructions


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


class Discriminator(nn.Module):
    def __init__(self, z=32):
        super(Discriminator, self).__init__()

        def linear_block(in_feat, out_feat, dropout=None):
            layers = [nn.Linear(in_feat, out_feat)]
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            return layers

        self.z = z
        self.discriminator = nn.Sequential(
            *linear_block(self.z, 1024, dropout=0.5),
            *linear_block(1024, 1024, dropout=0.5),
            *linear_block(1024, 1024, dropout=0.5),
            *linear_block(1024, 1024, dropout=0.5),
            *linear_block(1024, 1024, dropout=0.5),
            *linear_block(1024, 2),
        )

    def forward(self, latents, *args):
        return self.discriminator(latents)


class CustomFactorVAETraining(VAETraining):
    """Training setup for FactorVAE - VAE with disentangled latent space."""
    def __init__(self, encoder, decoder, discriminator, data,
               optimizer=torch.optim.Adam,
               max_epochs=50,
               batch_size=128,
               gamma=100,
               device="cpu",
               network_name="network",
               **kwargs):
        """Training setup for FactorVAE - VAE with disentangled latent space.
        Args:
            encoder (nn.Module): encoder neural network.
            decoder (nn.Module): decoder neural network.
            discriminator (nn.Module): auxiliary discriminator
              for approximation of latent space total correlation.
            data (Dataset): dataset providing training data.
            kwargs (dict): keyword arguments for generic VAE training.
        """
        super(CustomFactorVAETraining, self).__init__(
          encoder, decoder, data,
          optimizer=optimizer,
          max_epochs=max_epochs,
          batch_size=batch_size,
          device=device,
          network_name=network_name,
          **kwargs
        )
        self.gamma = gamma
        self.discriminator = discriminator.to(device)
        self.discriminator_optimizer = optimizer(
          self.discriminator.parameters(),
          lr=1e-4
        )

    def divergence_loss(self, normal_parameters, tc_parameters):
        tc_loss = vl.tc_encoder_loss(*tc_parameters)
        div_loss = vl.normal_kl_loss(*normal_parameters)
        result = div_loss - self.gamma * tc_loss #TODO: is the plus correct here??
        return result, tc_loss, div_loss
#     def sample(self, mean, logvar):
#         distribution = Normal(mean, torch.exp(0.5 * logvar))
#         sample = distribution.rsample()
#         return sample
    def loss(self, normal_parameters, tc_parameters,
           reconstruction, target):
        ce = self.reconstruction_loss(reconstruction, target)
        fl, tc_loss, div_loss = self.divergence_loss(normal_parameters, tc_parameters)
        loss_val = ce + fl
        self.current_losses["cross-entropy"] = float(ce)
        self.current_losses["vae"] = float(fl)
        self.current_losses["total-correlation"] = float(tc_loss)
        self.current_losses["kullback-leibler"] = float(div_loss)
        return loss_val
    def run_networks(self, data, *args):
        _, mean, logvar = self.encoder(data, *args)
        sample = self.sample(mean, logvar)
        reconstruction = self.decoder(sample, *args)
        return (mean, logvar), reconstruction, sample
    def step(self, data):
        data = to_device(data, self.device)
        sample_data, shuffle_data = data[:data.size(0) // 2], data[data.size(0) // 2:]
        normal_parameters, reconstruction, sample = self.run_networks(data)
        loss_val = self.loss(normal_parameters, (self.discriminator, sample), reconstruction, data)
        if self.verbose:
            for loss_name in self.current_losses:
                loss_float = self.current_losses[loss_name]
                self.writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
        self.writer.add_scalar("total loss", float(loss_val), self.step_id)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()
        _, _, shuffle_sample = self.run_networks(shuffle_data)
        self.discriminator_optimizer.zero_grad()
        discriminator_loss = vl.tc_discriminator_loss(
          self.discriminator,
          sample.detach(),
          shuffle_sample.detach()
        )
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        self.writer.add_scalar("discriminator-loss", float(discriminator_loss), self.step_id)
        self.each_step()

class OdirVAETraining(CustomFactorVAETraining):
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

        imgs = torch.zeros_like(reconstructions[0:50:10])

        for i in range(0, 50, 10):
            imgs[i] = F.sigmoid(reconstructions[i])

        if self.step_id % 20 == 0:
            self.writer.add_images("target", data[0:50:10], self.step_id)
            self.writer.add_images("reconstruction", imgs, self.step_id)
        return mean, logvar, reconstructions, data





if __name__ == '__main__':
    fake_imgs = torch.randn((4, 3, 192, 188))
    encoder = Encoder()
    encoder(fake_imgs)
    fake_latent_vectors = torch.randn((10, 32))
    decoder = Decoder()
    decoder(fake_latent_vectors)
