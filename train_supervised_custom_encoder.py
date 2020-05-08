# Standard Python Modules
from __future__ import print_function, division
import os
import sys
import time
import argparse
import copy

# pndas, plt, numpy, and other
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Standard Torch Modules
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# torchvision
import torchvision
from torchvision import datasets, transforms as T
from torchvision import utils as vutils
from torchvision.utils import save_image
from torchvision import models

# Torchsupport
import torchsupport
from torchsupport.training.vae import VAETraining
from torchsupport.training.training import SupervisedTraining

# Augmentor
import Augmentor

# skimage.sklearn
from skimage import io, transform as skT

# other
from tensorboardX import SummaryWriter
from torchsummary import summary

# for Resnet
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

# Own Modules
from utils.utils import setup
from utils.get_mean_std import get_mean_std

################### definitions #########################
def reconstructFileName(s):
    """give it something like '9_left_rot_foo.jpg'
    and it returns '9_left.jpg', that was the original
    file name before augmentation."""
    s = s.replace(".jpg", "")
    s = s.split("_")
    s = s[:2]
    s = "_".join(s)
    s += ".jpg"
    return s


def addAugmentationAnnotations(imdir, csv_file):
    """Give it the path for the image dir and a the
    path for the annotations.
    The image dir contains all the original images plus the
    augmentations, whcih by convention are names
    'original_name_augmentationsuffix.jpg'
    The function adds entries for the augmentations which are
    duplications of the original image labels (except their name/imdex)"""
    df = pd.read_csv(csv_file, header=0, sep="\t", index_col="Fundus Image")
    # df = df.loc[df.index.intersection(names)]
    names = os.listdir(imdir)
    onames = [reconstructFileName(f) for f in names]
    df2 = df.loc[onames]
    df2.index = names
    return df2


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def add_slash(path):
    if path[-1] != "/":
        return path + "/"
    else:
        return path


def bce(predict, target):
    result = F.binary_cross_entropy_with_logits(predict, target, reduction="sum")
    return result


class RetinnSuperVisedDataset(Dataset):
    """Expects to get a path for a directory containing images,
    and path for a csv file with annotations. Every image file
    should have annotations but there may be more annotations then
    images in the csv. Optionally also provide transform to perform
    on the images"""

    def __init__(self, imdir, csv_file, transform=T.ToTensor()):
        # df = pd.read_csv(csv_file, header=0,
        #        index_col='Fundus Image',
        #        sep='\t' )
        self.imdir = imdir
        self.csv_file = csv_file
        self.transform = transform
        self.imnames = os.listdir(self.imdir)
        self.labels = addAugmentationAnnotations(imdir, csv_file)

    def __len__(self):
        l = os.listdir(self.imdir)
        return len(l)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.imnames[idx]
        image = io.imread(os.path.abspath(self.imdir) + "/" + img_name)
        if self.transform != None:
            image = self.transform(image)
        cols = self.labels.columns[5:13]  # N,D,...,O
        labels = self.labels.loc[img_name, cols]
        labels = torch.Tensor(labels)
        sample = {"image": image, "labels": labels, "name": img_name}
        return sample["image"], sample["labels"]

def train_model(
    model,
    dataloaders, #already determined batch_sized in the dataloader
    optimizer, #already determined parameters and lr in this
    num_epochs=31,
    criterion=nn.BCEWithLogitsLoss(reduction="sum"),
    is_inception=False,
    report_interval=19,
):
    """
           The train_model function handles the training and validation of a given
        model. As input, it takes a PyTorch model, a dictionary of dataloaders, a loss
        function, an optimizer, a specified number of epochs to train and validate for,
        and a boolean flag for when the model is an Inception model. The is_inception
        flag is used to accomodate the Inception v3 model, as that architecture uses an
        auxiliary output and the overall model loss respects both the auxiliary output
        and the final output, as described here. The function trains for the specified
        number of epochs and after each epoch runs a full validation step. It also keeps
        track of the best performing model (in terms of validation accuracy), and at the
        end of training returns the best performing model. After each epoch, the
        training and validation accuracies are printed.
    """
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    lossarray = []
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            report_counter = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                report_counter += 1
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == "train":
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if report_counter % report_interval == 0:
                            lossarray.append(loss.item())

                    #_, preds = torch.max(outputs, 1)
                    preds = torch.max(outputs, torch.ones(8).to(device)).to(device)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #running_corrects += torch.sum(preds == labels)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, lossarray



#################### Plot Function #####################################
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import auc

#temp_save_dir = './temp_save/resnet101_pretrained2/'
temp_save_dir = './temp_save/custum_encoder/'

def plot_figures(temp_save_dir,
        csv_file,
        trainfolder,
        testfolder,
        encoder_name,
        device='cpu'):
#temp_save_dir = './temp_save/'
    """testing encoder.py from henrik's branch
    with a resnet"""
    lossarray = torch.load(temp_save_dir + 'lossarray.list', map_location='cpu')
    device='cpu'
    hist = torch.load(temp_save_dir + 'history.list', map_location='cpu')
    input_labels = torch.load(temp_save_dir + 'input_labels', map_location='cpu')
    output_labels = torch.load(temp_save_dir + 'output_labels', map_location='cpu')

    #trainfolder = test_dir
    #testfolder = valid_dir
    #csv_file = 'data/odir-training.csv'

    # plt.io()

    figures_dir = temp_save_dir + "figures/"
    #figures_dir = "/data/analysis/ag-reils/ag-reils-shared-students/yiftach/vae_for_retinal_images/data/supervised"
    #encoder_name = "resnet101"
    os.makedirs(figures_dir + f'/{encoder_name}', exist_ok=True)

    csv_df = pd.read_csv(csv_file, header=0, index_col="Fundus Image", sep="\t")

    diagnoses = {
    "N": "normal fundus",
    "D": "proliferative retinopathy",
    "G": "glaucoma",
    "C": "cataract",
    "A": "age related macular degeneration",
    "H": "hypertensive retinopathy",
    "M": "myopia",
    "O": "other diagnosis"
    }

    number_of_diagnoses = len(diagnoses)
    diagnoses_list = list(diagnoses.keys())
    diagnoses_list.extend(["Patient Sex"])

    x = np.arange(len(lossarray))

    spl = UnivariateSpline(x, lossarray)

    plt.title("Loss-Curve-Resnet101-pretrained", fontsize=16, fontweight='bold')
    plt.plot(x, lossarray, '-y')
    plt.plot(x, spl(x), '-r')
    plt.savefig(f'{figures_dir}/{encoder_name}_loss_curve.png')
    # plt.show()
    plt.close()

    outputs = output_labels.to(device="cpu").detach().numpy()
    targets = input_labels.float().numpy()
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'indigo', 'darkgreen', 'firebrick', 'sienna',
                  'red', 'limegreen']

    tpr = dict()  # Sensitivity/False Positive Rate
    fpr = dict()   # True Positive Rate / (1-Specifity)
    auc = dict()


    # A "micro-average": quantifying score on all classes jointly
    tpr["micro"], fpr["micro"], _ = roc_curve(targets.ravel(), outputs.ravel())
    auc["micro"] = roc_auc_score(targets.ravel(), outputs.ravel(), average='micro')
    print('AUC score, micro-averaged over all classes: {0:0.2f}'.format(auc['micro']))


    plt.figure()
    plt.step(tpr['micro'], fpr['micro'], where='post')
    plt.xlabel('False Positive Rate / Sensitivity', fontsize=11)
    plt.ylabel('True Negative Rate / (1 - Specifity)', fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'AUC score, micro-averaged over all classes: AP={0:0.2f}'
            .format(auc["micro"]), fontsize=13, fontweight='bold')
    #plt.show()
    plt.savefig(f'{figures_dir}/{encoder_name}/ROC_curve_micro_averaged.png')
    plt.close()


    # Plot of all classes ('macro')
    for i in range(number_of_diagnoses):
        tpr[i], fpr[i], _ = roc_curve(targets[:, i], outputs[:, i])
        try:
            auc[i] = roc_auc_score(targets[:, i], outputs[:, i])
        except ValueError:
            print(i, diagnoses_list[i], targets[:,i], outputs[:,i])

    plt.figure(figsize=(7, 9))
    lines = []
    labels = []

    l, = plt.plot(tpr["micro"], fpr["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-averaged ROC-AUC = {0:0.2f})'.format(auc["micro"]))

    for i, color in zip(range(number_of_diagnoses), colors):
        if i in auc.keys():
            l, = plt.plot(tpr[i], fpr[i], color=color, lw=0.5)
            lines.append(l)
            if diagnoses_list[i] != "Patient Sex":
                labels.append('ROC for class {0} (ROC-AUC = {1:0.2f})'
                              ''.format(diagnoses[diagnoses_list[i]], auc[i]))
            else:
                labels.append('ROC for class {0} (ROC-AUC = {1:0.2f})'
                              ''.format(diagnoses_list[i], auc[i]))


    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlabel('False Positive Rate / Sensitivity', fontsize=11)
    plt.ylabel('True Negative Rate / (1 - Specifity)', fontsize=11)
    plt.title('ROC curve of all features', fontsize=13, fontweight='bold')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=8))
    plt.savefig(f'{figures_dir}/{encoder_name}/ROC_curve_of_all_features.png')
    #plt.show()
    plt.close()

    # Precision-Recall Plots
    precision = dict()
    recall = dict()
    average_precision = dict()

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(targets.ravel(), outputs.ravel())
    average_precision["micro"] = average_precision_score(targets, outputs, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]), fontsize=13, fontweight='bold')
    plt.show()
    plt.savefig(f'{figures_dir}/{encoder_name}/PR_curve_micro_averaged.jpg')
    plt.close()

    # Plot of all classes ('macro')
    for i in range(number_of_diagnoses):
        precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs[:, i])
        average_precision[i] = average_precision_score(targets[:, i], outputs[:, i])

    plt.figure(figsize=(7, 9))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (average precision = {0:0.2f}'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(number_of_diagnoses), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=0.5)
        lines.append(l)
        if diagnoses_list[i] != "Patient Sex":
            labels.append('Precision-recall for class {0} (AP = {1:0.2f})'
                          ''.format(diagnoses[diagnoses_list[i]], average_precision[i]))
        else:
            labels.append('Precision-recall for class {0} (AP = {1:0.2f})'
                          ''.format(diagnoses_list[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Precision-Recall curve of all features')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=9))
    plt.show()
    plt.savefig(f'{figures_dir}/{encoder_name}/PR_curve_of_all_features.jpg')
    plt.close()


################## Custom Encoder ##############################

class Encoder(nn.Module):
    def __init__(self, number_of_features, z=32):
        # Incoming image has shape e.g. 192x188x3
        super(Encoder, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_max_pooling=0):
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
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
            *conv_block(32, 64, kernel_size=5, padding=2),  # New "Image" Size:  48x47
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
            nn.Linear(64, number_of_features),
            nn.BatchNorm1d(number_of_features),
            nn.Sigmoid()
            # *linear_block(64, 8, negative_slope=0.0)
        )

        self.mean = nn.Linear(z, z)
        self.logvar = nn.Linear(z, z)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        features = features.view(-1, np.prod(features.shape[1:]))
        features = self.linear_layers(features)
        # mean = self.mean(features)
        # logvar = self.logvar(features)
        return features  # , mean, logvar

########################## Runtime #####

csv_file = "../retina/outputs/supervised_sets/odir-training.csv"
test_dir = "../retina/data/processed/ODIR_Training_224x224/images/"
valid_dir= "../retina/data/processed/ODIR_Testing_224x224/images/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
zdim = 8
#input_size=224 #for resnet
input_size=(192,188) #for henrik's encoder
num_epochs=31

#plt.ion()
#plt.show()


#mytransform = T.Compose([T.ToTensor(), normalize])
#mytransform = T.Compose([T.ToPILImage(),
#    T.CenterCrop(input_size),
#    T.ToTensor(), normalize])
mytransform = T.Compose([T.ToPILImage(),
#    T.CenterCrop(input_size),
    T.Resize(input_size),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

#prepare model
#model = models.resnet101(pretrained=False)

#model = models.resnet101(pretrained=True)
#model.requires_grad_(False)
#model.layer4.requires_grad_(True)
#model.fc.requires_grad_(True)

#model.fc = nn.Linear(model.fc.in_features, zdim, bias=True)

model = Encoder(number_of_features=8, z=8)




#test and validation datasets
test_dataset = RetinnSuperVisedDataset(test_dir, csv_file, transform=mytransform)
valid_dataset = RetinnSuperVisedDataset(valid_dir, csv_file, transform=mytransform)

# test datasets
#temp_path = "./smalloutputdir"
#temp_dataset = RetinnSuperVisedDataset(temp_path,
#    csv_file="./data/odir-training.csv",  transform=mytransform)
#
#xx,yy = temp_dataset.__getitem__(1)
#xx
#yy
###############3

#put them in a dictionary:
image_datasets = {'train' : test_dataset, 'val' : valid_dataset}

dataloaders_dict = {x : DataLoader(image_datasets[x],
    batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train',
    'val']}

model.to(device)

feature_extract=False #for untrained model
#feature_extract=True #for pretrained resnet

params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
# set non_default lr here if desired
optimizer_ft = optim.Adam(params_to_update)

criterion=nn.BCEWithLogitsLoss(reduction='sum')
num_epochs = 31
#num_epochs = 61

model, hist, lossarray = train_model(model, dataloaders_dict, optimizer_ft,
        num_epochs=num_epochs, is_inception=False)


#temp save:
#temp_save_dir = './temp_save/resnet101_pretrained2/'
temp_save_dir = './temp_save/custum_encoder/'

os.makedirs(temp_save_dir, exist_ok=True)
torch.save(model.state_dict(), temp_save_dir + 'model_state.dict')
torch.save(hist, temp_save_dir + 'history.list')
torch.save(lossarray, temp_save_dir + 'lossarray.list')

vloader = DataLoader(valid_dataset, batch_size=79)

# get the outputs
model.eval()
device2 = 'cpu'
model.to(device2)
output_labels = []
input_labels = []

for inputs, labels in vloader:
    inputs = inputs.to(device2)
    labels = labels.to(device2)
    outputs = model(inputs)
    output_labels.append(outputs)
    output_labels = [torch.cat(output_labels, 0)]
    input_labels.append(labels)
    input_labels = [torch.cat(input_labels, 0)]

#get a tensor of all the labels/outputs for the plot
input_labels = input_labels[0]
output_labels = output_labels[0]

torch.save(input_labels, temp_save_dir + 'input_labels')
torch.save(output_labels, temp_save_dir + 'output_labels')

# plots
plot_figures(temp_save_dir=temp_save_dir,
        csv_file = 'data/odir-training.csv',
        trainfolder = test_dir,
        testfolder = valid_dir,
        encoder_name = "custom encoder",
        device='cpu')

