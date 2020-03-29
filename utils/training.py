"""
Add the Training (TorchSupport-Training API) and loss functions here.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torchsupport.training.vae import VAETraining

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

    #        return (data[0].unsqueeze(0),)

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
            bias=False,
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
                out_channels=36,
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
                in_channels=36,
                out_channels=39,
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
                in_channels=39,
                out_channels=42,
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
            nn.Linear(42 * 5 * 7, z),
            nn.ReLU(),
        )

        #            nn.Conv2d(
        #                in_channels=3, out_channels=6, kernel_size=5, stride=1, bias=False
        #            ),  # 252x316
        #            nn.ReLU(),
        #            nn.MaxPool2d(kernel_size=4, stride=4),  # 63x79
        #            nn.Conv2d(
        #                in_channels=6, out_channels=12, kernel_size=4, stride=1, bias=False
        #            ),  # 60x76
        #            nn.ReLU(),
        #            nn.MaxPool2d(kernel_size=4, stride=4),  # 15x19
        #            nn.Conv2d(
        #                in_channels=12, out_channels=24, kernel_size=4, stride=1, bias=False
        #            ),  # 12x16
        #            nn.ReLU(),
        #            nn.MaxPool2d(kernel_size=4, stride=4),  # 24x3x4
        #            nn.Flatten(1),  # 24*3*4
        #            nn.Linear(24 * 3 * 4, 32),
        #            nn.ReLU(),  # 32
        #        )

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
        #        self.decoder = nn.Sequential(
        #            nn.Linear(z, 256 * 320 * 3),
        #            nn.ReLU(),
        #            #            nn.Sigmoid()
        #            #            nn.Linear(z, 128),
        #            #            nn.ReLU(),
        #            #            nn.Linear(128, 256),
        #            #            nn.ReLU(),
        #            #            nn.Linear(256, 28 * 32),
        #            #            nn.ReLU(),
        #            #            nn.Linear(28 * 32, 36 * 48),
        #            #            nn.ReLU(),
        #            #            nn.Linear(36 * 48, 72 * 96),
        #            #            nn.ReLU(),
        #            #            nn.Linear(72 * 96, 156 * 212),
        #            #            nn.ReLU(),
        #            #            nn.Linear(156 * 212, 256 * 320)
        #        )

        def linear_block(in_feat, out_feat, normalize=True, dropout=None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(
                nn.BatchNorm1d(out_feat)
            )  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.linear_blocks = nn.Sequential(
            *linear_block(z, 64, normalize=False),
            *linear_block(64, 256),
            *linear_block(256, 320, dropout=0.5),
            *linear_block(320, 576, dropout=0.5),  # 44*6*6 = 1584
            nn.ReLU(),
        )

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
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
                nn.Upsample(mode="bilinear", scale_factor=2),
            ]

        self.conv_layers = nn.Sequential(
            *conv_block(64, 54, padding=1),
            *conv_block(54, 36, padding=1),
            *conv_block(36, 24, padding=1),
            *conv_block(24, 8, padding=1),
            *conv_block(8, 3, kernel_size=5, padding=2),
            # nn.UpsamplingNearest2d(size=(192, 188)),  # The wished size
            nn.UpsamplingNearest2d(size=(256, 320)),  # The wished size
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 8),
            nn.ReLU(),
            nn.Linear(256 * 8, 256 * 24),
            nn.ReLU(),
            nn.Linear(256 * 24, 256 * 64),
            nn.ReLU(),
            nn.Linear(256 * 64, 256 * 128),
            nn.ReLU(),
            nn.Linear(256 * 128, 256 * 320),
            nn.ReLU(),
            nn.Linear(256 * 320, 256 * 320 * 3),
        )

    def forward(self, sample):
        return self.decoder(sample).view(-1, 3, 256, 320)


#        return self.decoder(sample).view(-1, 1, 256, 320)


class OdirVAETraining(VAETraining):
    def run_networks(self, data, *args):
        mean, logvar, reconstructions, data = super().run_networks(data, *args)
        if self.step_id % 4 == 0:
            self.writer.add_image("target", data[0], self.step_id)
            self.writer.add_image("reconstruction", reconstructions[0], self.step_id)
        return mean, logvar, reconstructions, data
