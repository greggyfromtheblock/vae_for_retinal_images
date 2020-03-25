<<<<<<< HEAD
"""
This preprocessing script trims the black edges of the images and keeps it so that we have
a minimum black border on the edges of the image.
Also has a --resize flag to resize the images to a certain dimension if needed
"""

from __future__ import division, print_function

import argparse
import os
# Ignore warnings
import warnings

import numpy as np
from skimage import io
from skimage.transform import resize

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Database Image Size Scanner')
parser.add_argument('imgdir', type=str, default=None, metavar='image_dir',
                    help="""The path to the directory which contains
                    the imgages""")
parser.add_argument('outdir', type=str, default=None, metavar='output_dir',
                    help="""The path of the new directory where
                    they will be saved""")
parser.add_argument('--resize', type=int, nargs=2, default=[0,0], metavar='resize shape',
                    help="""if resize is desired
                    use this option and give it 2 ints""")
args = parser.parse_args() 


# Correcting the possibly missing slash.
# It's lame but works for this little purpose
if args.imgdir[-1] != '/':
    args.imgdir += '/'
if args.outdir[-1] != '/':
    args.outdir += '/'

prefix = "preptest"
outdir = "preptest/"
imdir = 'odir/ODIR-5K_Training_Dataset/'


outdir = args.outdir
imdir = args.imgdir

os.makedirs(outdir,exist_ok=True)


def crop(img, r=0):
    """Trims the black margins out of the image
    The originaland returned images are rgb"""
    ts = (img != 0).sum(axis=1) != 0
    ts = ts.sum(axis=1) != 0
    img = img[ts]
    ts = (img != 0).sum(axis=0) != 0
    ts = ts.sum(axis=1) != 0
    img = img[:,ts,:]
    if r != [0,0] and r != (0,0) and r !=0:
        img = resize(img, r, anti_aliasing=True)
    return img

# Generate the standardized image set
for f in os.listdir(imdir):
    fname = f.replace(".jpg", "")
    image = io.imread(imdir + f)
    image = crop(image, args.resize)
    image = image.astype(np.uint8)
    io.imsave(outdir + f , image)
=======
from decode_diagnostics_keywords import decode_d_k
from tqdm import tqdm
from preprocessing_methods import (
    trim_image_rgb,
    rotate,
    find_optimal_image_size_and_extend_db,
)
import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
import numpy as np
import sys
import argparse




# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# Example command line run from within the utils folder:
# python preprocessing.py ../../data/raw/ODIR_Training_Images/ process/images/ ../../data/train/ ../../data/raw/
# the excel file should be in ../../data/raw and the raw images
# should be inside a subfolder of it. 

if __name__ == "__main__":
    """
    Preprocessing Steps:
    Trim the black margins out of the image.
    Find the 'optimal' image size, means: 
        calculate the ratio of width and heigth of the cropped images,
        resize images to minimal image size, if it is close to the avg. ratio,
        hereby is purposed to avoid transforming the circle shape of retinas to ellipses by resizing the images
    Subsequently, the augmentation step follows:
    Flip images, Rotate those images whose retinas are ca complete circles 
    Example command line, run from within 'utils':
    python preprocessing.py ../../data/raw/ODIR_Training_Images/ process/images/ ../../data/train/ ../../data/raw/
    """
    parser = argparse.ArgumentParser(
            description="""Preprocessing""")
    parser.add_argument('rawdir', type=str, default=None,
        metavar='raw_image_dir',
                    help="""The path to the directory which contains
                    the imgages""")
    parser.add_argument('outdir_name', type=str, default=None,
        metavar='output_dir_name',
                    help="""The name of the new directory where 
                    they will be saved""")
    parser.add_argument('dir', type=str, default=None,
        metavar='root_output_dir',
                    help="""The path of the new directory where 
                    they will be saved""")
    parser.add_argument('root_dir', type=str, default=None,
        metavar='root_input_dir',
                    help="""The path of the directory where 
                    the the odir folder is stored""")
    args = parser.parse_args()

    ddir = "/home/henrik/PycharmProjects/vae_for_retinal_images/data/"
    odir = "odir/ODIR-5K_Training_Dataset/"
    outdir = "processed/train/"

    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return(path)

    ddir = add_slash(args.dir)
    odir = add_slash(args.rawdir)
    outdir = add_slash(args.outdir_name)
    rootdir = add_slash(args.root_dir)


    os.makedirs(ddir + outdir, exist_ok=True)

    print("Start cropping...")
    for f in tqdm(os.listdir(ddir + odir)):
        # Crop image
        trim_image_rgb(f, ddir + odir, ddir + outdir)
    print("Finished cropping...")

    print("Start finding optimal image size and extend db...")
    opt_w, opt_h = find_optimal_image_size_and_extend_db(
            db = rootdir, 
            imdir = ddir + outdir, out="extended.tsv")
#            imdir = ddir + outdir, out="odir/extended.tsv")
    print("Finished finding optimal image size and extend db...")

    print("Start resizing and data augmentation...")
    for f in tqdm(os.listdir(ddir + outdir)):
        fname = f.replace(".jpg", "")
        image = io.imread(ddir + outdir + f)

        # Resize image
        image = resize(image, output_shape=(opt_w, opt_h))

        # save image under processed data
        io.imsave(ddir + outdir + f, img_as_ubyte(image))

        # flip image
        image_flipped = np.fliplr(image)
        io.imsave(ddir + outdir + fname + "_flipped.jpg", img_as_ubyte(image_flipped))

        # rotate image and save it
        rotate(image, ddir + outdir, fname)
        rotate(image_flipped, ddir + outdir, fname + "_flipped")

    print("Finished resizing and data augmentation...")

    print("Decode diagnostics keywords...")
    #decode_d_k(ddir)
    decode_d_k(path=rootdir, output_file = "extended.csv" )
    print("Finished decoding diagnostics keywords...")
>>>>>>> e3c358b253f5d73c7291b1bda835560c5b8d279b
