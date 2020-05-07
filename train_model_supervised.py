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

# from utils.training_supervised import (
#    RetinnSuperVisedDataset,
#    SupervisedTraining,
# )


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


class SupervisedCustomTraining:
    """Give it a dataset, parameters and shit"""

    def __init__(
        self,
        dataset,
        net,
        network_name="myNetwork",
        figures_dir="myNetworkFiguresAreAwesomeMehtaWorldPeace",
        optimizer=optim.Adam,
        maxpoch=10,
        device="cpu",
        batch_size=64,
        learning_rate=5e-5,
        criterion=nn.BCEWithLogitsLoss(),
    ):
        self.dataset = dataset
        self.figures_dir = figures_dir
        self.batch_size = batch_size
        self.network_name = network_name
        self.figures_dir = figures_dir
        self.maxpoch = maxpoch
        self.learning_rate = (learning_rate,)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net = net.to(device=self.device)
        self.criterion = criterion.to(device=self.device)
        self.optimizer = optimizer(self.net.parameters(),)
        print("loaded model and parameters")
        print(model)


def train_model(
    model,
    dataloaders, #already determined batch_sized in the dataloader
    optimizer, #already determined parameters and lr in this
    num_epochs=25,
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
            #running_corrects = 0
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



### Tests #####

#csv_file = "../preptest2/odir_training_annotations.csv"
#test_dir = "../preptest2/test_images/images/"
#valid_dir = "../preptest2/valid_images/images/"
#csv_file = "../retina/outputs/supervised_sets/odir-training.csv"
test_dir = "../retina/outputs/supervised_sets/training_images/images/"
valid_dir= "../retina/outputs/supervised_sets/validation_images/images/"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
zdim = 8
num_epochs=35

plt.ion()

#df = pd.read_csv(csv_file, header=0, index_col="Fundus Image", sep="\t")
#df[df["anterior"] == 1]
#df[df["no fundus"] == 1]

plt.show()

input_size=224 #in case we crop the image, that's the standard resnet size

mytransform = T.Compose([T.ToTensor(), normalize])

mytransform = T.Compose([T.ToPILImage(),
    T.CenterCrop(input_size),
    T.ToTensor(), normalize])

#prepare model
#model = models.resnet101(pretrained=False)
model = models.resnet101(pretrained=True)
model.requires_grad_(False)
model.layer4.requires_grad_(True)
model.fc.requires_grad_(True)
model.fc = nn.Linear(model.fc.in_features, zdim, bias=True)

#test and validation datasets
test_dataset = RetinnSuperVisedDataset(test_dir, csv_file, transform=mytransform)
valid_dataset = RetinnSuperVisedDataset(valid_dir, csv_file, transform=mytransform)

#xx,yy = test_dataset.__getitem__(1)
#xx
#yy
#put them in a dictionary:
image_datasets = {'train' : test_dataset, 'val' : valid_dataset}

dataloaders_dict = {x : DataLoader(image_datasets[x],
    batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train',
    'val']}

model.to(device)

feature_extract=False #irrelevant for us

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
optimizer_ft = optim.Adam(params_to_update)

criterion=nn.BCEWithLogitsLoss(reduction='sum')

model, hist = train_model(model, dataloaders_dict, optimizer_ft,
        num_epochs=num_epochs, is_inception=False)

#temp save:
temp_save_dir = './temp_save/'

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


#################
#Henrik encoder.py 
####################
def test_run():
    """testing encoder.py from henrik's branch
    with a resnet"""

    #trainfolder = sys.argv[1]
    trainfolder = test_dir
    #testfolder = sys.argv[2]
    testfolder = valid_dir
    #csv_file = sys.argv[3]     
    csv_file = csv_file     

    figures_dir = "/data/analysis/ag-reils/ag-reils-shared-students/yiftach/vae_for_retinal_images/data/supervised"
    encoder_name = "resnet101"
    os.makedirs(figures_dir + f'/{encoder_name}', exist_ok=True)

    #device = "cuda:5" if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.cuda.clear_memory_allocated()
    torch.cuda.empty_cache()
    # torch.cuda.memory_stats(device)

    #csv_df = pd.read_csv(csv_file, sep='\t')
    csv_df = pd.read_csv(csv_file, header=0, index_col="Fundus Image", sep="\t")
    #csv_df = addAugmentationAnnotations(trainfolder, csv_file)

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
    angles = [x for x in range(-22, -9)]
    angles.extend([x for x in range(10, 22 + 1)])
    angles.extend([x for x in range(-9, 10)])
    print("\nPossible Angles: {}\n".format(angles))

    print('Finished Training\nTrainingtime: %d sec' % (time.perf_counter() - start))
    x = np.arange(len(lossarray))
    spl = UnivariateSpline(x, lossarray)
    plt.title("Loss-Curve", fontsize=16, fontweight='bold')
    plt.plot(x, lossarray, '-y')
    plt.plot(x, spl(x), '-r')
    plt.savefig(f'{figures_dir}/{encoder_name}_loss_curve.png')
    # plt.show()
    plt.close()

    
    
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

    outputs = outputs.to(device="cpu").detach().numpy()
    targets = targets.float().numpy()
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
    plt.show()
    plt.savefig(f'{figures_dir}/{encoder_name}/ROC_curve_micro_averaged.png')
    plt.close()

    # Plot of all classes ('macro')
    for i in range(number_of_diagnoses + 1):
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

    for i, color in zip(range(number_of_diagnoses + 1), colors):
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
    plt.show()
    plt.close()

    # Precision-Recall Plots
    precision = dict()
    recall = dict()
    average_precision = dict()
    from sklearn.metrics import auc

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
    for i in range(number_of_diagnoses + 1):
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

    for i, color in zip(range(number_of_diagnoses + 1), colors):
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
    os.system(f"cp encoder.py {figures_dir}/{encoder_name}/encoder.py")
