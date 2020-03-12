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
import random

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
parser.add_argument('--resize', type=int, nargs=2, default=[0,0],
        metavar='resize shape',
                    help="""if resize is desired 
                    use this option and give it 2 ints""")
parser.add_argument('--flip', action='store_true',
                    help="""if this flag is on 
                    the script creates an addtional flipped version
                    of the images""")
parser.add_argument('--rotate', type=float, default=0,
        metavar='rotate',
                    help="""if this value is given 
                    the script creates an addtional rotated version
                    of the images. The rotation is random between 0
                    and the provided argument""")
args = parser.parse_args()

#print(args, args.db, args.dir, args.output)
#print(args.resize)

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
    """Trimms the black margins out of the image
    The returned image is in gray scale"""
    ts = (img != 0).sum(axis=1) != 0
    img = img[ts].transpose()
    tss = (img != 0).sum(axis=1) != 0
    img = img[tss].transpose()
    return img

def trim_image_rgb(img, r=0):
    """Trimms the black margins out of the image
    The originaland returned images are rgb"""
    ts = (img != 0).sum(axis=1) != 0
    ts = ts.sum(axis=1) != 0
    img = img[ts]
    ts = (img != 0).sum(axis=0) != 0
    ts = ts.sum(axis=1) != 0
    img = img[:,ts,:]
    #img[:,:,0] = trim_image(img[:,:,0])
    #img[:,:,1] = trim_image(img[:,:,1])
    #img[:,:,2] = trim_image(img[:,:,2])
    if r != [0,0] and r != (0,0) and r !=0:
        img = resize(img, r, anti_aliasing=True)
    return img

# Generate the standardized image set
for f in os.listdir(imdir):
    #image = rgb2gray(io.imread(imdir + f))
    fname = f.replace(".jpg", "")
    image = io.imread(imdir + f)
    image = trim_image_rgb(image, args.resize)
    image = image.astype(np.uint8)
    io.imsave(outdir + f, image) 
    if args.flip:
        image_flipped = np.fliplr(image)
        io.imsave(outdir + fname + "_flipped.jpg", image_flipped) 
    if args.rotate > 0:
        rot = args.rotate*random.random()
        image_rotated = transform.rotate(image, rot)
        io.imsave(outdir + fname + "_rotated.jpg", image_rotated) 
    #break

#img_uint8 = img.astype(np.uint8)
