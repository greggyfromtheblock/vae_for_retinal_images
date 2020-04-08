"""
Add the Training (TorchSupport-Training API) and loss functions here.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torchsupport.training.vae import VAETraining
import torch.nn.functional as F
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
        return (data,)

    def __len__(self):
        return len(self.data)
        pass


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


class Encoder(nn.Module):
    def __init__(self, z=32):
        super(Encoder, self).__init__()
        self.z = z

        def conv_block(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            padding_max_pooling=0,
            relu=True,
            batchnorm=False,
            maxpool=False,
            dropout=0,
            kernel_maxpool=2,
            stride_maxpool=2,
            bias=True,
        ):
            block = []
            block.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
            if relu:
                block.append(nn.ReLU())
            if dropout > 0:
                block.append(nn.Dropout2d(dropout))
            if batchnorm:
                block.append(nn.BatchNorm2d(out_channels))
            if maxpool:
                block.append(
                    nn.MaxPool2d(
                        kernel_size=kernel_maxpool,
                        stride=stride_maxpool,
                        padding=padding_max_pooling,
                    )
                )
            return block

        self.encoder = nn.Sequential(
            *conv_block(
                in_channels=3,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=False,
                dropout=0.05,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 6x256x320
            *conv_block(
                in_channels=6,
                out_channels=9,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=True,
                dropout=0,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 9x128x160
            *conv_block(
                in_channels=9,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=False,
                dropout=0.05,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 12x128x160
            *conv_block(
                in_channels=12,
                out_channels=15,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=True,
                dropout=0,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 15x64x80
            *conv_block(
                in_channels=15,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=False,
                dropout=0.05,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 18x46x80
            *conv_block(
                in_channels=18,
                out_channels=21,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=True,
                dropout=0,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 21x32x40
            *conv_block(
                in_channels=21,
                out_channels=24,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=False,
                dropout=0.05,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 24x32x40
            *conv_block(
                in_channels=24,
                out_channels=27,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=True,
                dropout=0,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 27x16x20
            *conv_block(
                in_channels=27,
                out_channels=30,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=True,
                maxpool=False,
                dropout=0.05,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 30x16x20
            *conv_block(
                in_channels=30,
                out_channels=33,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_max_pooling=0,
                relu=True,
                batchnorm=False,
                maxpool=True,
                dropout=0,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 33x8x10
            *conv_block(
                in_channels=33,
                out_channels=46,
                kernel_size=2,
                stride=1,
                padding=0,
                padding_max_pooling=0,
                relu=True,
                batchnorm=False,
                maxpool=False,
                dropout=0,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 36x7x9
            *conv_block(
                in_channels=46,
                out_channels=59,
                kernel_size=2,
                stride=1,
                padding=0,
                padding_max_pooling=0,
                relu=True,
                batchnorm=False,
                maxpool=False,
                dropout=0,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 39x6x8
            *conv_block(
                in_channels=59,
                out_channels=72,
                kernel_size=2,
                stride=1,
                padding=0,
                padding_max_pooling=0,
                relu=True,
                batchnorm=False,
                maxpool=False,
                dropout=0,
                kernel_maxpool=2,
                stride_maxpool=2,
            ),  # 42x5x7
            nn.Flatten(1),
            nn.Linear(72 * 5 * 7, z),
            nn.ReLU(),
        )
        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        # inputs = inputs.view(inputs.size(0), -1)
        features = self.encoder(inputs)
        mean = self.mean(features)
        logvar = self.logvar(features)
        return features, mean, logvar


class Decoder(nn.Module):
    def __init__(self, z=32):
        super(Decoder, self).__init__()
        self.z = z

        def linear_block(in_feat, out_feat, normalize=True, dropout=None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(
                nn.BatchNorm1d(out_feat)
            )  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.152, inplace=True))
            return layers

        self.linear_blocks = nn.Sequential(
            *linear_block(z, 64, normalize=False),
            *linear_block(64, 256),
            *linear_block(256, 320, dropout=0.1),
            *linear_block(320, 576, dropout=0.05),
            *linear_block(576, 1280),  # 64x4x5=1280
            nn.ReLU(),
        )

        def conv_block(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            scale_factor=2,
        ):
            return [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.Upsample(mode="bilinear", scale_factor=scale_factor),
            ]

        self.conv_layers = nn.Sequential(  # 64x4x5
            *conv_block(64, 60, padding=0, kernel_size=2),  # 60x10x12
            *conv_block(60, 56, padding=0, kernel_size=2),  # 56x22x26
            *conv_block(56, 52, padding=0, kernel_size=2),  # 52x46x54
            *conv_block(52, 32, padding=1, kernel_size=3, scale_factor=1.5),  # 32x69x81
            *conv_block(
                32, 18, padding=1, kernel_size=3, scale_factor=1.5
            ),  # 18x103x121
            *conv_block(
                18, 10, padding=1, kernel_size=3, scale_factor=1.5
            ),  # 10x154x181
            *conv_block(10, 7, padding=1, kernel_size=3, scale_factor=1.5),  # 7x231x271
            *conv_block(7, 3, padding=1, kernel_size=3, scale_factor=1.1),  # 7x231x271
            nn.UpsamplingNearest2d(size=(256, 320)),  # The wished size
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
        )

    def forward(self, sample):
        #        return self.decoder(sample).view(-1, 3, 256, 320)
        dec = torch.reshape(self.linear_blocks(sample),
                (sample.shape[0], 64, 4, 5))
        # print(dec.shape)
        reconstructions = self.conv_layers(dec)
        print(reconstructions.shape)
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
        #what is that?
        if self.epoch != self.epoch_id:
            self.epoch = self.epoch_id
            print("%i-Epoch" % (self.epoch_id+1))

        if self.step_id % 4 == 0:
            self.writer.add_image("target", data[0], self.step_id)
            self.writer.add_image("reconstruction",
                    nn.functional.sigmoid(reconstructions[0]), self.step_id)
            print("output shape: ", reconstructions[0].shape)
        return mean, logvar, reconstructions, data
