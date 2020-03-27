"""
Add the Training (TorchSupport-Training API) and loss functions here.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsupport.training.vae import VAETraining
# from vae import VAETraining
from tensorboardX import SummaryWriter
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
            return [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=padding_max_pooling)]

        self.conv_layers = nn.Sequential(
            # Formula of new "Image" Size: (origanal_size - kernel_size + 2 * amount_of_padding)//stride + 1
            *conv_block(3, 8, kernel_size=5, stride=1, padding=2),  # (192-5+2*2)//1 + 1 = 192  > Max-Pooling: 190/2=96
            # -> (188-5+2*2)//1 + 1 = 188  --> Max-Pooling: 188/2 = 94
            *conv_block(8, 16, kernel_size=5, padding=2),   # New "Image" Size:  48x47
            *conv_block(16, 24, padding=1),  # New "Image" Size:  24x23
            *conv_block(24, 36, padding=0, padding_max_pooling=1),  # New "Image" Size:  11x10
            *conv_block(36, 64, padding=0, padding_max_pooling=0),  # New "Image" Size:  5x4
            # *conv_block(54, 64),   # New "Image" Size:  2*2
        )

        def linear_block(in_feat, out_feat, normalize=True, dropout=None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(nn.BatchNorm1d(out_feat))  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.linear_layers = nn.Sequential(
            # output_channels: 64; 3 x 3 from image dimension; 64*3*3 = 576
            *linear_block(64 * 5 * 4, 512, normalize=False, dropout=None),
            *linear_block(512, 256, dropout=None),
            *linear_block(256, 128),
            *linear_block(128, 64),
            nn.Linear(64, z),
            nn.BatchNorm1d(z),
            nn.ReLU()
        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        # print(features.shape)
        features = features.view(-1, self.num_flat_features(features))
        # print(features.shape)

        features = self.linear_layers(features)
        # print(8,features.shape)
        mean = self.mean(features)
        logvar = self.logvar(features)
        return features, mean, logvar

    @staticmethod
    def num_flat_features(x):
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
            *linear_block(64, 128),
            *linear_block(128, 256, dropout=None),
            *linear_block(256, 512, dropout=None),
            *linear_block(512, 1280, dropout=None),  # 5*4*64
            nn.Sigmoid()
        )

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, scale_factor=None, size=None, mode='bilinear'):

            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(),
                      nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(),
                      nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU()]

            scale_factor and layers.append(nn.Upsample(mode=mode, scale_factor=scale_factor))
            size and layers.append(nn.Upsample(mode=mode, size=size))
            return layers

        self.conv_layers = nn.Sequential(
            *conv_block(64, 48, padding=1, size=(11, 10)),
            *conv_block(48, 32, padding=1, size=(24, 23), mode='nearest'),
            *conv_block(32, 24, padding=1, size=(48, 47)),
            *conv_block(24, 8, padding=1, scale_factor=2, mode='nearest'),
            *conv_block(8, 3, kernel_size=5, padding=2, scale_factor=2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.BatchNorm2d(3),
        )

    def forward(self, latent_vector):
        dec = torch.reshape(self.linear_blocks(latent_vector), (latent_vector.shape[0], 64, 5, 4))
        # print(dec.shape)
        reconstructions = self.conv_layers(dec)
        # print(7, reconstructions.shape)
        return reconstructions


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
            print("%i-Epoch" % self.epoch_id)
        if self.step_id % 10 == 0:
            self.writer.add_images("target", data[0:50:10], self.step_id)
            self.writer.add_images("reconstruction", reconstructions[0:50:10], self.step_id)
        return mean, logvar, reconstructions, data


if __name__ == '__main__':
    fake_imgs = torch.randn((4, 3, 192, 188))
    encoder = Encoder()
    encoder(fake_imgs)
    fake_latent_vector = torch.randn((10, 32))
    decoder = Decoder()
    decoder(fake_latent_vector)




