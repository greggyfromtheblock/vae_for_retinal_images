"""
Add the Training (TorchSupport-Training API) and loss functions here.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchsupport.training.vae import VAETraining
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

""" def normalize(image):
    print(image.shape)
    for i, each_dim in enumerate(image):
        image[i] = (each_dim - each_dim.min()) / (each_dim.max() - each_dim.min())
    return image """


def normalize(image):
    (image - image.min()) / (image.max() - image.min())
    return


class Encoder(nn.Module):
    def __init__(self, z=32):
        # Incoming image has shape e.g. 192x188x3
        super(Encoder, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_max_pooling=0):
            return [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=padding_max_pooling)]

        self.conv_layers = nn.Sequential(
            # Formula of new "Image" Size: (origanal_size - kernel_size + 2 * amount_of_padding)//stride + 1
            *conv_block(3, 8, kernel_size=3, stride=1, padding=1),  # (192-3+2*1)//1 + 1 = 192  > Max-Pooling: 190/2=96
            # -> (188-3+2*1)//1 + 1 = 188  --> Max-Pooling: 188/2 = 94
            *conv_block(8, 16, kernel_size=3, padding=1),   # New "Image" Size:  48x44
            *conv_block(16, 24, padding=1),     # New "Image" Size:  24x22
            *conv_block(24, 36, padding=1, padding_max_pooling=1),   # New "Image" Size:  12x12
            *conv_block(36, 54, padding=1),   # New "Image" Size:  6x6
            *conv_block(54, 64, padding=1),   # New "Image" Size:  3*3
        )

        def linear_block(in_feat, out_feat, normalize=True, dropout=None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(nn.BatchNorm1d(out_feat))  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.linear_layers = nn.Sequential(
            # output_channels: 64; 3 x 3 from image dimension; 64*3*3 = 576
            *linear_block(64 * 3 * 3, 320, normalize=False, dropout=0.5),
            *linear_block(320, 256, dropout=0.5),
            *linear_block(256, 128),
            *linear_block(128, 64),
            nn.Linear(64, z),
            nn.ReLU()
        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        features = features.view(-1, self.num_flat_features(features))
        features = self.linear_layers(features)

        mean = self.mean(features)
        logvar = self.logvar(features)
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
            *linear_block(256, 320, dropout=0.5),
            *linear_block(320, 576, dropout=0.5),   # 44*6*6 = 1584
            nn.ReLU()
        )

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
            return [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Upsample(mode='bilinear', scale_factor=2)]

        self.conv_layers = nn.Sequential(
            *conv_block(64, 54, padding=1),
            *conv_block(54, 36, padding=1),
            *conv_block(36, 24, padding=1),
            *conv_block(24, 8,  padding=1),
            *conv_block(8, 3, kernel_size=5, padding=2),
            nn.UpsamplingNearest2d(size=(192, 188)),  # The wished size
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
        )

    def forward(self, latent_vector):
        dec = torch.reshape(self.linear_blocks(latent_vector), (latent_vector.shape[0], 64, 3, 3))
        reconstructions = self.conv_layers(dec)
        return reconstructions


class OdirVAETraining(VAETraining):
    def __init__(self, encoder, decoder, data, path_prefix, net_name,
                 # alpha=0.25, beta=0.5, m=120,
                 # optimizer=torch.optim.Adam,
                 # optimizer_kwargs=None,
                 **kwargs):
        super(OdirVAETraining, self).__init__(
            encoder, decoder, data,
            #  optimizer=optimizer,
            #  optimizer_kwargs=optimizer_kwargs,
            **kwargs
        )
        self.checkpoint_path = f"{path_prefix}/{net_name}/{net_name}-checkpoint"
        self.writer = SummaryWriter(f"{path_prefix}/{net_name}/")
        self.epoch = 0

    def run_networks(self, data, *args):
        mean, logvar, reconstructions, data = super().run_networks(data, *args)
        # for i in range(0,50,10):
        #    data[i] = normalize(data[i])
        not self.epoch and print("%i-Epoch" % self.epoch_id)    # print zeroth epoch

        if self.epoch != self.epoch_id:
            self.epoch = self.epoch_id
            print("%i-Epoch" % self.epoch_id)
        if self.step_id % 10 == 0:
            self.writer.add_image("target", normalize(data[0]), self.step_id)
            self.writer.add_image("reconstruction", normalize(reconstructions[0]), self.step_id)
        return mean, logvar, reconstructions, data








