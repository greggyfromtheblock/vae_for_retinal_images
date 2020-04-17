import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from scipy.interpolate import UnivariateSpline
from torchvision import datasets, transforms
import numpy as np
import torch
import os
from tqdm import tqdm
import pandas as pd
from utils.training import VAEDataset

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def add_slash(path):
    if path[-1] != '/':
        return path + "/"
    else:
        return(path)


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
            *conv_block(32, 64, kernel_size=5, padding=2),   # New "Image" Size:  48x47
            *conv_block(64, 128, padding=1),  # New "Image" Size:  24x23
            *conv_block(128, 256, padding=0, padding_max_pooling=1),  # New "Image" Size:  11x10
            *conv_block(256, 512, padding=0, padding_max_pooling=0),  # New "Image" Size:  5x4
            *conv_block(512, 256, padding=0, padding_max_pooling=1),  # New "Image" Size:  2x2
        )

        def linear_block(in_feat, out_feat, normalize=True, dropout=None, negative_slope=1e-2):
            layers = [nn.Linear(in_feat, out_feat)]
            normalize and layers.append(nn.BatchNorm1d(out_feat))  # It's the same as: if normalize: append...
            dropout and layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            return layers

        self.linear_layers = nn.Sequential(
            *linear_block(1024, 512, normalize=True, dropout=0.5),
            *linear_block(512, 256, dropout=0.3),
            *linear_block(256, 128),
            *linear_block(128, 64),
            *linear_block(64, 8, negative_slope=0.0)
        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        features = features.view(-1, np.prod(features.shape[1:]))
        features = self.linear_layers(features)
        # mean = self.mean(features)<
        # logvar = self.logvar(features)
        return features   # , mean, logvar


if __name__ == '__main__':

    trainfolder = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/processed/training/PALM_and_ODIR/Images"
    testfolder = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/processed/training/n-augmentation_6_maxdegree_20_resize_192_188_grayscale_0/ODIR/"
    csv_file = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv"
    
    print("\nLoad Data as Tensors...")
    img_dataset = datasets.ImageFolder(
        os.path.dirname(os.path.dirname(trainfolder)),
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )
    data = VAEDataset(img_dataset)
    print("\nSize of the dataset: {}\nShape of the single tensors: {}".format(len(data), data[0][0].shape))

    csv_df = pd.read_csv(csv_file, sep='\t')

    diagnoses = {
        "N": "normal fundus",
        "D": "proliferative retinopathy",
        "G": "glaucoma",
        "C": "cataract",
        "A": "age related macular degeneration",
        "H": "hypertensive retinopathy",
        "M": "myopia",
        # "ant": "anterior segment",
        # "no": "no fundus image",
    }
    number_of_diagnoses = len(diagnoses)
    data_size = len(data)

    targets = np.zeros((data_size, number_of_diagnoses + 1), dtype=np.float16)
    age_targets = np.zeros(data_size, dtype=np.uint8)

    angles = [x for x in range(-22, -9)]
    angles.extend([x for x in range(10, 22 + 1)])
    angles.extend([x for x in range(-9, 10)])
    print("\nPossible Angles: {}\n".format(angles))

    print("\nBuild targets...")
    marker = None
    for i, jpg in tqdm(enumerate(os.listdir(trainfolder))):
        if jpg == '.snakemake_timestamp':
            marker = True
            continue
        jpg = jpg.replace("_flipped", "")

        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]

        diagnoses_list = list(diagnoses.keys())
        diagnoses_list.extend(["Patient Sex"])
        for j, feature in enumerate(diagnoses_list):
            if not marker:
                if feature == "N":
                    targets[i][j] = not csv_df.iloc[row_number].at[feature]
                else:
                    if feature == "Patient Sex":
                        targets[i][j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                    else:
                        targets[i][j] = csv_df.iloc[row_number].at[feature]
            else:
                if feature == "N":
                    targets[i - 1][j] = not csv_df.iloc[row_number].at[feature]
                else:
                    if feature == "Patient Sex":
                        targets[i - 1][j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                    else:
                        targets[i - 1][j] = csv_df.iloc[row_number].at[feature]
        if marker:
            age_targets[i - 1] = csv_df.iloc[row_number].at["Patient Age"]
        else:
            age_targets[i - 1] = csv_df.iloc[row_number].at["Patient Age"]

    net = Encoder()

    # Train the network
    n_epochs = 60
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    lossarray = []

    def calc_batch_size(batch_size=8):
        global data_size
        if data_size % batch_size == 0:
            return batch_size
        if batch_size == 3:
            return batch_size
        else:
            return calc_batch_size(batch_size - 1)

    # calculate batch_size
    batch_size = calc_batch_size(batch_size=64)

    # Train network
    start = time.perf_counter()
    for n in range(n_epochs):
        running_loss = 0.0

        inputs = torch.zeros((batch_size, *data[0][0].shape))
        labels = torch.zeros((batch_size, number_of_diagnoses + 1), dtype=torch.float)
        d_mod_b = data_size % batch_size
        targets = torch.Tensor(targets).float()
        for i in tqdm(range(0, data_size, batch_size)):
            if (i + batch_size) < data_size:
                for j in range(batch_size):
                    inputs[j] = data[i + j][0]
                    labels[j] = targets[i+j]
            elif d_mod_b != 0:
                # for uncompleted last batch
                labels = torch.zeros((d_mod_b, number_of_diagnoses + 1))
                inputs = torch.zeros((d_mod_b, *data[0][0].shape))
                for j in range(d_mod_b):
                    inputs[j] = data[i + j][0]
                    labels[j] = targets[i+j]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % batch_size == batch_size - 1:  # print every 10 mini-batches
            # print('[%d, %5d] loss: %.3f' % (n + 1, i + 1, running_loss / batch_size))
            lossarray.append(loss.item())
            running_loss = 0.0

    print('Finished Training\nTrainingtime: %d sec' % (time.perf_counter() - start))

    x = np.arange(len(lossarray))
    spl = UnivariateSpline(x, lossarray)
    plt.plot(x, lossarray, '-y')
    plt.plot(x, spl(x), '-r')
    # plt.show()

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    ########################################
    #           Test network               #
    ########################################

    print("\nLoad Data as Tensors...")
    img_dataset = datasets.ImageFolder(
        os.path.dirname(os.path.dirname(testfolder)),
        transform=transforms.Compose([transforms.ToTensor(), normalize])
    )
    data = VAEDataset(img_dataset)
    print("\nSize of the test dataset: {}\nShape of the single tensors: {}".format(len(data), data[0][0].shape))

    data_size = len(data)
    targets = np.zeros((data_size, number_of_diagnoses + 1), dtype=np.float16)

    print("\nBuild targets...")
    marker = None
    for i, jpg in tqdm(enumerate(os.listdir(testfolder))):
        if jpg == '.snakemake_timestamp':
            marker = True
            continue
        jpg = jpg.replace("_flipped", "")

        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")

        row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]

        diagnoses_list = list(diagnoses.keys())
        diagnoses_list.extend(["Patient Sex"])
        for j, feature in enumerate(diagnoses_list):
            if not marker:
                if feature == "N":
                    targets[i][j] = not csv_df.iloc[row_number].at[feature]
                else:
                    if feature == "Patient Sex":
                        targets[i][j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                    else:
                        targets[i][j] = csv_df.iloc[row_number].at[feature]
            else:
                if feature == "N":
                    targets[i - 1][j] = not csv_df.iloc[row_number].at[feature]
                else:
                    if feature == "Patient Sex":
                        targets[i - 1][j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                    else:
                        targets[i - 1][j] = csv_df.iloc[row_number].at[feature]

    # Test the network
    inputs = torch.zeros((batch_size, *data[0][0].shape))
    labels = torch.zeros((batch_size, number_of_diagnoses + 1), dtype=torch.float)
    d_mod_b = data_size % batch_size
    targets = torch.Tensor(targets).int()

    correct = 0
    total = 0
    # with torch.no_grad():
    print(targets[0,0])
    with torch.no_grad():
        for i in tqdm(range(0, data_size, batch_size)):
            if (i + batch_size) < data_size:
                for j in range(batch_size):
                    inputs[j] = data[i + j][0]
                    labels[j] = targets[i + j]
            elif d_mod_b != 0:
                # for uncompleted last batch
                labels = torch.zeros((d_mod_b, number_of_diagnoses + 1))
                inputs = torch.zeros((d_mod_b, *data[0][0].shape))
                for j in range(d_mod_b):
                    inputs[j] = data[i + j][0]
                    labels[j] = targets[i + j]

            outputs = torch.round(net(inputs))
            total += labels.size(0) * (number_of_diagnoses+1)
            correct += (outputs == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))





