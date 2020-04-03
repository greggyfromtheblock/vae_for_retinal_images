import os
import sys
import argparse
import warnings
from skimage import io, img_as_ubyte
from skimage.transform import resize
import numpy as np
from preprocessing_methods import (
    trim_image_rgb,
    rotate,
    find_optimal_image_size_and_extend_db,
)
from tqdm import tqdm

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
    Flip images, Rotate those images whose retinas are complete circles 
    Example command line, run from within 'utils':
    python preprocessing.py ../../data/raw/ODIR_Training_Images/ process/images/ ../../data/train/ ../../data/raw/
    """
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument(
        "rawdir",
        type=str,
        default=None,
        metavar="raw_image_dir",
        help="""The path to the directory which contains
                    the imgages""",
    )
    parser.add_argument(
        "outdir",
        type=str,
        default=None,
        metavar="output_dir",
        help="""The full path of the new directory where 
                    they will be saved""",
    )
    parser.add_argument(
        "--n_augmentation",
        type=int,
        default=0,
        help="""Number of Augmented images per image""",
    )
    parser.add_argument(
        "--max_rotation",
        type=int,
        default=0,
        help="""Max rotation degree +- for the images, for example if you pass 10 to
                        this argument then the function will pick {aug_per_image} random values from
                        the range -10 to 10""",
    )
    args = parser.parse_args()

    def add_slash(path):
        if path[-1] != "/":
            return path + "/"
        else:
            return path
    rawdir = os.path.abspath(args.rawdir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("Start cropping...")
    #for f in tqdm(os.listdir(ddir + odir)):
    for f in tqdm(os.listdir(rawdir)):
        # Crop image
        #trim_image_rgb(f, ddir + odir, ddir + outdir)
        trim_image_rgb(f, add_slash(rawdir), add_slash(outdir))
    print("Finished cropping...")

    print("Start finding optimal image size and extend db...")
    #opt_w, opt_h = find_optimal_image_size_and_extend_db(imdir=ddir + outdir)
    opt_w, opt_h = find_optimal_image_size_and_extend_db(imdir=outdir)
    #            imdir = ddir + outdir, out="odir/extended.tsv")
    print("Finished finding optimal image size and extend db...")

    print("Start resizing and data augmentation...")
    #for f in tqdm(os.listdir(ddir + outdir)):
    for f in tqdm(os.listdir(outdir)):
        fname = f.replace(".jpg", "")
        #image = io.imread(ddir + outdir + f)
        image = io.imread(outdir + '/' + f)

        # Resize image
        image = resize(image, output_shape=(opt_w, opt_h))

        # save image under processed data
        #io.imsave(ddir + outdir + f, img_as_ubyte(image))
        io.imsave(outdir + '/' + f, img_as_ubyte(image))

        # flip image
        image_flipped = np.fliplr(image)
        #io.imsave(ddir + outdir + fname + "_flipped.jpg", img_as_ubyte(image_flipped))
        io.imsave(outdir + '/' + fname + "_flipped.jpg", img_as_ubyte(image_flipped))

        # rotate image and save it
        #rotate(image, ddir + outdir, fname)
        #rotate(image,outdir, fname)
        #rotate(image_flipped, ddir + outdir, fname + "_flipped")
        #rotate(image_flipped, outdir, fname + "_flipped")
       
    # rotate image and save it
        # Rotation might be broken so 
        # Comment it out for now
        # old rotate:
        #rotate(image, ddir + outdir, fname)
        #rotate(image_flipped, ddir + outdir, fname + "_flipped")
        # new rotate (gregg):
        #rotate(image, outdir, fname, args.n_augmentation, args.max_rotation)
        #rotate(
        #    image_flipped,
        #    outdir,
        #    fname + "_flipped",
        #    args.n_augmentation,
        #    args.max_rotation,
        #)

        

    print("Finished resizing and data augmentation...")
