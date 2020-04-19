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

# for Resnet
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

from torchsummary import summary

import torchvision.models as models

from collections import OrderedDict

############## Resnet ###################################################


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
        )  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

conv = conv3x3(in_channels=32, out_channels=64)
print(conv)
del conv


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=1,
        downsampling=1,
        conv=conv3x3,
        *args,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = (
            nn.Sequential(
                OrderedDict(
                    {
                        "conv": nn.Conv2d(
                            self.in_channels,
                            self.expanded_channels,
                            kernel_size=1,
                            stride=self.downsampling,
                            bias=False,
                        ),
                        "bn": nn.BatchNorm2d(self.expanded_channels),
                    }
                )
            )
            if self.should_apply_shortcut
            else None
        )

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(
        OrderedDict(
            {
                "conv": conv(in_channels, out_channels, *args, **kwargs),
                "bn": nn.BatchNorm2d(out_channels),
            }
        )
    )


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(
                self.in_channels,
                self.out_channels,
                conv=self.conv,
                bias=False,
                stride=self.downsampling,
            ),
            activation(),
            conv_bn(
                self.out_channels, self.expanded_channels, conv=self.conv, bias=False
            ),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation(),
            conv_bn(
                self.out_channels,
                self.out_channels,
                self.conv,
                kernel_size=3,
                stride=self.downsampling,
            ),
            activation(),
            conv_bn(
                self.out_channels, self.expanded_channels, self.conv, kernel_size=1
            ),
        )


class ResNetLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs
    ):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(
                in_channels, out_channels, *args, **kwargs, downsampling=downsampling
            ),
            *[
                block(
                    out_channels * block.expansion,
                    out_channels,
                    downsampling=1,
                    *args,
                    **kwargs,
                )
                for _ in range(n - 1)
            ],
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(
        self,
        in_channels=3,
        blocks_sizes=[64, 128, 256, 512],
        deepths=[2, 2, 2, 2],
        activation=nn.ReLU,
        block=ResNetBasicBlock,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.blocks_sizes[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList(
            [
                ResNetLayer(
                    blocks_sizes[0],
                    blocks_sizes[0],
                    n=deepths[0],
                    activation=activation,
                    block=block,
                    *args,
                    **kwargs,
                ),
                *[
                    ResNetLayer(
                        in_channels * block.expansion,
                        out_channels,
                        n=n,
                        activation=activation,
                        block=block,
                        *args,
                        **kwargs,
                    )
                    for (in_channels, out_channels), n in zip(
                        self.in_out_block_sizes, deepths[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

#    def forward(self, x):
#        x = self.gate(x)
#        for block in self.blocks:
#            x = block(x)
#        mean = self.mean(x)
#        logvar = self.logvar(x)
#        return x, mean, logvar

#    def forward(self, inputs):
#        # inputs = inputs.view(inputs.size(0), -1)
#        features = self.encoder(inputs)
#        mean = self.mean(features)
#        logvar = self.logvar(features)
#        return features, mean, logvar


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])


def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])


def resnet50(in_channels, n_classes):
    return ResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3]
    )


def resnet101(in_channels, n_classes):
    return ResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3]
    )


def resnet152(in_channels, n_classes):
    return ResNet(
        in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3]
    )


#ResidualBlock(32, 64)
#dummy = torch.ones((1, 1, 1, 1))
#block = ResidualBlock(1, 64)
#block(dummy)
#ResNetResidualBlock(32, 64)
#conv_bn(3, 3, nn.Conv2d, kernel_size=3)
#dummy = torch.ones((1, 32, 256, 320))
#block = ResNetBasicBlock(32, 64)
#block(dummy).shape
#print(block)
#dummy = torch.ones((1, 32, 10, 10))
#block = ResNetBottleNeckBlock(32, 64)
#block(dummy).shape
#print(block)
#
#dummy = torch.ones((1, 32, 48, 48))
#
#dummy2 = torch.ones((10, 64, 48, 48))
#
#layer = ResNetLayer(64, 128, block=ResNetBasicBlock, n=3)
#
#####?!#### layer(dummy).shape
#layer(dummy2).shape
#
#layer
#
#
#model = resnet101(3, 32)
#
#summary(model, (3, 256, 320))
#
#dummy3 = torch.ones((10, 3, 256, 320))
#
#x = model.encoder(dummy3)
#x.shape
#y = model.decoder(x)
#y.shape

############################################################


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

        model = resnet101(3, z) #3: rgb input channels, 32: latent space dim

        self.encoder = model.forward()
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
        dec = torch.reshape(self.linear_blocks(sample), (sample.shape[0], 64, 4, 5))
        # print(dec.shape)
        reconstructions = self.conv_layers(dec)
        print(reconstructions.shape)
        return reconstructions


class OdirVAETraining(VAETraining):
    def __init__(
        self,
        encoder,
        decoder,
        data,
        path_prefix,
        network_name,
        # alpha=0.25, beta=0.5, m=120,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr": 5e-5},
        **kwargs,
    ):
        super(OdirVAETraining, self).__init__(
            encoder,
            decoder,
            data,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )
        self.checkpoint_path = f"{path_prefix}/{network_name}/{network_name}-checkpoint"
        self.writer = SummaryWriter(f"{path_prefix}/{network_name}/")
        self.epoch = None

    def run_networks(self, data, *args):
        mean, logvar, reconstructions, data = super().run_networks(data, *args)
        # what is that?
        if self.epoch != self.epoch_id:
            self.epoch = self.epoch_id
            print("%i-Epoch" % (self.epoch_id + 1))

        if self.step_id % 4 == 0:
            self.writer.add_image("target", data[0], self.step_id)
            self.writer.add_image(
                "reconstruction",
                nn.functional.sigmoid(reconstructions[0]),
                self.step_id,
            )
            print("output shape: ", reconstructions[0].shape)
        return mean, logvar, reconstructions, data
