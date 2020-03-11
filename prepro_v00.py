#!/home/zelhar/miniconda3/envs/test/bin/python
from __future__ import print_function, division
from skimage import data, color, io, transform
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL as pil
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Database Image Size Scanner')
parser.add_argument('dir', type=str, default=None,
        metavar='image_dir',
                    help="""The path to the directory which contains
                    the imgages""")
parser.add_argument('outdir', type=str, default=None,
        metavar='output_dir',
                    help="""The path of the new directory where 
                    they will be saved""")
args = parser.parse_args()

#print(args, args.db, args.dir, args.output)

# Correcting the possibly missing slash.
#It's lame but works for this little purpose
if args.dir[-1] != '/':
    args.dir += '/'
if args.outdir[-1] != '/':
    args.outdir += '/'

prefix = "preptest"
outdir = "preptest/"
db = 'odir/ODIR-5K_Training_Annotations(Updated)_V2.xlsx'
imdir = 'odir/ODIR-5K_Training_Dataset/'
#out = 'odir/output.tsv'
#out = prefix + "_" + imdir

outdir = args.outdir
imdir = args.dir

os.makedirs(outdir, exist_ok=True)

def trim_image(img):
    """Trimms the black margins out of the image"""
    ts = (img != 0).sum(axis=1) != 0
    img = img[ts].transpose()
    tss = (img != 0).sum(axis=1) != 0
    img = img[tss].transpose()
    return img

for f in os.listdir(imdir):
    image = rgb2gray(io.imread(imdir + f))
    image = trim_image(image)
    io.imsave(outdir + f, image) 
    #print(f)
    #break

