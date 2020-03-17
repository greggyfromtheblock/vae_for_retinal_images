from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torchsupport
from torchsupport.training.vae import VAETraining

import Augmentor

# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")
#plt.ion()  # interactive mode

class RetinnDataset(Dataset):
    """Reinal Image Dataset. Input: path to the csv annotation file,
    and path for the image dir."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file, header=0,
                index_col=None,
                sep='\t' )
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.labels.loc[idx,
            'Left-Fundus'])
        image = io.imread(img_name)
        labels = self.labels.iloc[idx, 15:-4]
        labels = np.array([labels])
        labels = labels.astype("float").reshape(-1, 2)
        sample = {"image": image, "labels": labels}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

 class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for labels because for images,
        # x and y axes are axis 1 and 0 respectively
        return {'image': img, 'labels': labels}

class Crop(object):
    """Crops the black margins out of the image."""
    def __call__(self, sample):
        pass

class Augment(object):
    """Perform some augmentation operations on the sample.
    returned value: sample with same 'labels' but 'image' randomly
    transformed according to some input parameters of __init__."""
    pass

class Encoder(nn.Module):
    """VAE Encoder."""
    pass

class Decoder(nn.Module):
    """VAE Decoder."""
    pass

class Normalizer(object):
    """Image intensity normalizer. 
    returned value: sample with the same 'labels' but normalized
    'image'"""
    pass

class RetinnVAETraining(VAETraining):
    pass

### Tests ###
#xslfile = 'odir/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
csv_file = 'odir/odir_train_annot_complete_lr.csv'
imgdir = 'odir/ODIR-5K_Training_Dataset/'

df = pd.read_csv(csv_file, sep='\t', header=0, index_col=None)

transformations = transforms.Compose([Rescale((920,920)), 
    ToTensor()])

retinn_df = RetinnDataset(csv_file, imgdir, transformations)

dataset_loader = DataLoader(retinn_df, batch_size=5,
        shuffle=True)

plt.ion()


dataiter = iter(dataset_loader)

x,y = dataiter.next()

for i_batch, sample_batched in enumerate(dataset_loader):
    print(x)
    #print(i_batch)
    print(i_batch, sample_batched['image'],
          sample_batched['labels'])
    break

retinn_df.labels

retinn_df.root_dir
retinn_df.__len__()

retinn_df[0]['labels']
retinn_df[0]['image']

io.imshow(retinn_df[0]['image'].numpy().transpose((1,2,0)))

io.imshow(retinn_df[1]['image'].numpy().transpose((1,2,0)))

plt.pause(1)

plt.close()

plt.show()

