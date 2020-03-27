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

    def __len__(self):
        return len(self.data)
        pass


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


class Encoder(nn.Module):
    def __init__(self, z=32):
        # Incoming image has shape e.g. 192x188x3
        super(Encoder, self).__init__()

        def conv_block_nomp(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        ):
            return [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
            ]

        def conv_block(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            padding_max_pooling=0,
        ):
            return [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=padding_max_pooling),
            ]

#        self.conv_layers = nn.Sequential(
#            # Formula of new "Image" Size: (origanal_size - kernel_size + 2 * amount_of_padding)//stride + 1
#            *conv_block(
#                3, 8, kernel_size=3, stride=1, padding=1
#            ),  # (192-3+2*1)//1 + 1 = 192  > Max-Pooling: 190/2=96
#            # -> (188-3+2*1)//1 + 1 = 188  --> Max-Pooling: 188/2 = 94
#            *conv_block(8, 16, kernel_size=3, padding=1),  # New "Image" Size:  48x44
#            *conv_block(16, 24, padding=1),  # New "Image" Size:  24x22
#            *conv_block(
#                24, 36, padding=1, padding_max_pooling=1
#            ),  # New "Image" Size:  12x12
#            *conv_block(36, 54, padding=1),  # New "Image" Size:  6x6
#            *conv_block(54, 64, padding=1),  # New "Image" Size:  3*3
#        )

        self.conv_layers = nn.Sequential(
            *conv_block_nomp(
                3, 12, kernel_size=4, stride=1, padding=1
            ),  
            *conv_block_nomp(12, 16, kernel_size=4, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), #(16,125,157)
            *conv_block(16, 24, kernel_size=4), #(24,61,77)  
            *conv_block(
                24, 36, kernel_size=2
            ),  # (36,30,38)
            *conv_block(36, 54), #(54,14,18)  
            *conv_block(54, 64, padding=1), #(64,7,8) 
        )

        def linear_block(in_feat, out_feat, normalize=True, dropout=None):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(
                nn.BatchNorm1d(out_feat)
            )  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.125, inplace=True))
            return layers

        self.linear_layers = nn.Sequential(
            # output_channels: 64 x 7 x 8 
            *linear_block(64 * 7 * 8, 320, normalize=False,
                dropout=0.125),
            *linear_block(320, 256, dropout=0.5),
            *linear_block(256, 128),
            *linear_block(128, 64),
            nn.Linear(64, z),
            nn.ReLU(),
        )

#        self.linear_layers = nn.Sequential(
#            # output_channels: 64; 3 x 3 from image dimension; 64*3*3 = 576
#            *linear_block(64 * 3 * 3, 320, normalize=False, dropout=0.5),
#            *linear_block(320, 256, dropout=0.5),
#            *linear_block(256, 128),
#            *linear_block(128, 64),
#            nn.Linear(64, z),
#            nn.ReLU(),
#        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        # print(features.shape)
        features = features.view(-1, self.num_flat_features(features))
        # print(features.shape)
        features = self.linear_layers(features)

        mean = self.mean(features)
        logvar = self.logvar(features)
        # print(features.shape)
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
        self.z = z

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
            #nn.UpsamplingNearest2d(size=(192, 188)),  # The wished size
            nn.UpsamplingNearest2d(size=(256, 320)),  # The wished size
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, 256*8),
            nn.ReLU(),
            nn.Linear(256*8, 256*24),
            nn.ReLU(),
            nn.Linear(256*24, 256*64),
            nn.ReLU(),
            nn.Linear(256*64, 256*128),
            nn.ReLU(),
            nn.Linear(256*128, 256*320),
            nn.ReLU(),
            nn.Linear(256*320, 256*320*3))

    def forward(self, latent_vector):
        return self.decoder(latent_vector).view(-1,3,256,320)
#        dec = torch.reshape(
#            self.linear_blocks(latent_vector), (latent_vector.shape[0], 64, 3, 3)
#        )
#        # print(dec.shape)
#        reconstructions = self.conv_layers(dec)
#        print(reconstructions.shape)
#        return reconstructions


class OdirVAETraining(VAETraining):
    def run_networks(self, data, *args):
        mean, logvar, reconstructions, data = super().run_networks(data, *args)
        if self.step_id % 4 == 0:
            self.writer.add_image("target", data[0], self.step_id)
            self.writer.add_image("reconstruction", reconstructions[0], self.step_id)
        return mean, logvar, reconstructions, data


#        if self.step_id % 10 == 0:
#            self.writer.add_images("target", data[0:50:10], self.step_id)
#            self.writer.add_images(
#                "reconstruction", reconstructions[0:50:10], self.step_id
#            )
#        return mean, logvar, reconstructions, data


if __name__ == "__main__":
    # Test Encoder
    fake_imgs = torch.randn((10, 3, 192, 188))
    # print(fake_imgs.shape)
    encoder = Encoder()
    # encoder.forward(fake_imgs)

    # Test Decoder
    fake_latent_vector = torch.randn((10, 32))
    # print(233, fake_latent_vector.shape)
    decoder = Decoder()
    # decoder.forward(fake_latent_vector)
