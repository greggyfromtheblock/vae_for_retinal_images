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
import sys

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def add_slash(path):
    if path[-1] != '/':
        return path + "/"
    else:
        return(path)


def calc_batch_size(datasize, batch_size=128):
    for b_size in range(batch_size, 2, -1):
        if datasize % b_size == 0:
            return batch_size
        if b_size < 10 and datasize % b_size >= 2:
            return b_size


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
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
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
        return features   # , mean, logvar


if __name__ == '__main__':

    trainfolder = sys.argv[1] #  "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/processed/training/n-augmentation_6_maxdegree_20_resize_192_188_grayscale_0/ODIR/"
    testfolder = sys.argv[2] # "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/processed/testing/n-augmentation_6_maxdegree_20_resize_192_188_grayscale_0/ODIR/"
    csv_file = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/processed/annotations/ODIR_Annotations.csv"
    
    figures_dir = "/data/analysis/ag-reils/ag-reils-shared-students/henrik/vae_for_retinal_images/data/supervised"
    encoder_name = "deep_balanced"
    os.makedirs(figures_dir+f'/{encoder_name}', exist_ok=True)

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    # torch.cuda.clear_memory_allocated()
    torch.cuda.empty_cache()
    # torch.cuda.memory_stats(device)
    
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

    net = Encoder().to(device=device)
    print("Allocated memory: %s MiB" % torch.cuda.memory_allocated(device))
    
    # Train the network
    n_epochs = 2
    learning_rate = 5e-5
    criterion = nn.BCELoss().to(device=device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    lossarray = []

    # calculate batch_size
    batch_size = calc_batch_size(data_size, batch_size=128)

    # Train network
    start = time.perf_counter()
    targets = torch.Tensor(targets).float()
    
    print("Start Training")
    for n in tqdm(range(n_epochs)):
        running_loss = 0.0

        inputs = torch.zeros((batch_size, *data[0][0].shape)).cuda(device=device)
        labels = torch.zeros((batch_size, number_of_diagnoses + 1), dtype=torch.float).cuda(device=device)
        d_mod_b = data_size % batch_size
        for i in range(0, data_size, batch_size):
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
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % batch_size == batch_size - 1:
                print('[%d, %5d] loss: %.3f' % (n + 1, i + 1, running_loss / batch_size))
            lossarray.append(loss.item())
            running_loss = 0.0

    print('Finished Training\nTrainingtime: %d sec' % (time.perf_counter() - start))

    x = np.arange(len(lossarray))
    spl = UnivariateSpline(x, lossarray)
    plt.title("Loss-Curve", fontsize=16, fontweight='bold')
    plt.plot(x, lossarray, '-y')
    plt.plot(x, spl(x), '-r')
    plt.savefig(f'{figures_dir}/{encoder_name}_loss_curve.png')
    #plt.show()
    plt.close()

    PATH = f'{figures_dir}/{encoder_name}.pth'
    torch.save(net.state_dict(), PATH)

    ########################################
    #           Test network               #
    ########################################
    # torch.cuda.clear_memory_allocated()
    torch.cuda.empty_cache()
    # torch.cuda.memory_stats(device)

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
                if feature == "Patient Sex":
                    targets[i][j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                else:
                    targets[i][j] = csv_df.iloc[row_number].at[feature]
            else:
                if feature == "Patient Sex":
                    targets[i - 1][j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                else:
                    targets[i - 1][j] = csv_df.iloc[row_number].at[feature]

    # Test the network
    print("Start testing the network..")
    batch_size = calc_batch_size(data_size, batch_size=128)
    inputs = torch.zeros((batch_size, *data[0][0].shape))
    d_mod_b = data_size % batch_size
    targets = torch.Tensor(targets).int()
    
    print("Build predictions...")
    outputs = torch.zeros((data_size, number_of_diagnoses+1), device=device)
    for i in range(0, data_size, batch_size):
        if (i + batch_size) < data_size:
            for j in range(batch_size):
                inputs[j] = data[i + j][0]
            outputs[i:(i+batch_size)] = net(inputs.to(device)).detach()
        elif d_mod_b != 0:
            # for uncompleted last batch
            inputs = torch.zeros((d_mod_b, *data[0][0].shape))
            for j in range(d_mod_b):
                inputs[j] = data[i + j][0]
            outputs[i:(i+d_mod_b)] = net(inputs.to(device)).detach()
                                            
    # To measure the accuracy on the basic of the rounded outcome for each diagnosis could lead to a less
    # meaningful result. That's why this approach is deprecated and here outcommented.
    # In lieu thereof, a ROC curve is used.
    # The network has as an outcome a vector of floats with values between 0 and 1. The threshold to round up is
    # increased stepwise, starts with 0 until 1.
    # In every step we calculate the Sensitivity/True Positiv Rate (TRP) and the Specifity/True Negative Rate:
    # TRP = TP/(TP+FN)  and  FPR=FP/(TN+FP).
    # https://de.wikipedia.org/wiki/Beurteilung_eines_bin%C3%A4ren_Klassifikators#Sensitivit%C3%A4t_und_Falsch-Negativ-Rate
    # https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

    # roc_auc_score: Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    # average_auc_score: Compute average precision (AP) from prediction scores
    # AP = sum ((R_N - R_N_-1) * P_N)
    # AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the
    # increase in recall from the previous threshold used as the weight: where R_N and P_N are the precision and recall
    # at the n-th threshold. This implementation is not interpolated and is different from computing the area under the
    # precision-recall curve with the trapezoidal rule, which uses linear interpolation and can be too optimistic.
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

    # outputs = torch.round(net(inputs))
    # total += labels.size(0) * (number_of_diagnoses+1)
    # correct += (outputs == labels).sum().item()

    # correct = 0
    # total = 0
    # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    # ROC-Curve/AUC with sklearn:
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, auc

    try:
        outputs = outputs.to(device="cpu").detach().numpy()
        targets = targets.float().numpy()

        # calculate roc curves
        lr_fpr, lr_tpr, _ = roc_curve(targets, outputs)

        lr_auc = roc_auc_score(lr_fpr, lr_tpr, average='micro')
        # summarize scores
        print('Logistic: ROC AUC=%.3f' % (lr_auc))

        # plot the roc curve for the model
        plt.plot([0, 1], [0, 1], linestyle='--', c='darkorange')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        plt.grid()
        plt.title("ROC-Curve - Encoder", fontsize=16, fontweight='bold')
        plt.savefig(f'{figures_dir}/{encoder_name}_micro-averaged_ROC_curve_auc_{lr_auc}.png')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        plt.close()
    except ValueError:
        pass

    try:
        average_precision = average_precision_score(targets, outputs)
        print('2-class Precision-Recall curve: Average Precision={0:0.2f}'.format(average_precision))

        lr_precision, lr_recall, _ = precision_recall_curve(targets, outputs)
        lr_f1, lr_auc = f1_score(targets, outputs), auc(lr_recall, lr_precision)
        print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

        # plot the precision-recall curves
        plt.grid()
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        plt.title("Precision-Recall-Curve - ", fontsize=16, fontweight='bold')
        plt.savefig(f'{figures_dir}/{encoder_name}/micro-averaged_precision_recall_curve_auc_{lr_auc}_f1_{lr_f1}_ap_{average_precision}.png')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        plt.close()

    except ValueError:
        pass

    for i, diagnosis in enumerate(diagnoses_list):
        try:
            lr_auc = roc_auc_score(targets[:,i], outputs[:, i])
            # summarize scores
            print('Logistic: ROC AUC=%.3f' % (lr_auc))
            # calculate roc curves
            lr_fpr, lr_tpr, _ = roc_curve(targets[:,i], outputs[:,i])
            # plot the roc curve for the model
            plt.plot([0, 1], [0, 1], linestyle='--', c='darkorange')
            plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
            plt.grid()
            if diagnosis != "Patient Sex":
                plt.title("ROC-Curve - {}".format(diagnoses[diagnosis]), fontsize=16, fontweight='bold')
                plt.savefig(f'{figures_dir}/{encoder_name}/{diagnoses[diagnosis]}_ROC_curve.png')
            else:
                plt.title("ROC-Curve - Patient Sex - auc=%.3f" % lr_auc, fontsize=16, fontweight='bold')
                plt.savefig(f'{figures_dir}/{encoder_name}/Patient_Sex_ROC_curve_auc_{lr_auc}.png')
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.show()
            plt.close()

        except ValueError:
            pass

        try:
            average_precision = average_precision_score(targets[:, i], outputs[:, i])
            print('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

            lr_precision, lr_recall, _ = precision_recall_curve(targets[:, i], outputs[:, i])
            lr_f1, lr_auc = f1_score(targets[:, i], outputs[:, i]), auc(lr_recall, lr_precision)
            print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

            # plot the precision-recall curves
            plt.grid()
            plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
            if diagnosis != "Patient Sex":
                plt.title("Precision-Recall-Curve - %s f1=%.3f auc=%.3f" % (diagnoses[diagnosis], lr_f1, lr_auc), fontsize=16, fontweight='bold')
                plt.savefig(f'{figures_dir}/{encoder_name}/{diagnoses[diagnosis]}_precision_recall_curve_auc_{lr_auc}_f1_{lr_f1}_ap_{average_precision}.png')
            else:
                plt.title("Precision-Recall-Curve - Patient Sex", fontsize=16, fontweight='bold')
                plt.savefig(f'{figures_dir}/{encoder_name}/Patient_Sex_precision_recall_curve_auc_{lr_auc}_f1_{lr_f1}_ap_{average_precision}.png')
            # axis labels
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # show the legend
            plt.legend()
            # show the plot
            plt.show()
            plt.close()

        except ValueError:
            pass






