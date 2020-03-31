
import argparse
import os
import warnings

import numpy as np
from skimage import img_as_ubyte, io
from skimage.transform import resize
from tqdm import tqdm

from decode_diagnostics_keywords import decode_d_k
from preprocessing_methods import (find_optimal_image_size_and_extend_db,
                                   rotate, trim_image_rgb)

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    """
    Preprocessing Steps:
    Trim the black margins out of the image.
    Find the 'optimal' image size, means: 
        calculate the ratio of width and heigth of the cropped images,
        resize images to minimal image size, if it is close to the avg. ratio,
        hereby is purposed to avoid transforming the circle shape of retinas to ellipses by resizing the images
    Subsequently, the augmentation step follows:
    Flip images, Rotate those images whose retinas are complete circles 
    """
    parser = argparse.ArgumentParser(
        description="""Preprocessing""")
    parser.add_argument('imdir', type=str, default=None,
                        metavar='image_dir',
                        help="""The path to the directory which contains
                       the image folder. """)
    parser.add_argument('outdir', type=str, default=None,
                        metavar='target_dir',
                        help="""The path to the directory which should contain
                        processed augmented images.""")
    parser.add_argument('aug_per_image', type=int, default=0,
                        metavar='n_augmentation',
                        help="""Number of Augmented images per image""")
    parser.add_argument('max_rotation_angle', type=int, default=0,
                        metavar='maximum_rotation_angle',
                        help="""Max rotation degree +- for the images, for example if you pass 10 to
                        this argument then the function will pick {aug_per_image} random values from
                        the range -10 to 10""")
    args, unknown = parser.parse_known_args()

    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return(path)

    dir = add_slash(args.imdir)
    outdir = add_slash(args.outdir)

    # TODO: Uncouple resize function with preprocess annotations function on preprocessing_methods.py
    print("Start finding optimal image size and extend db...")
    opt_w, opt_h = find_optimal_image_size_and_extend_db(xlsx_dir, outdir)
    print("Finished finding optimal image size and extend db...")

    print("Start resizing and data augmentation...")
    for f in tqdm(os.listdir(outdir)):
        fname = f.replace(".jpg", "")
        image = io.imread(outdir + f)
        # Resize image
        image = resize(image, output_shape=(opt_w, opt_h))

        # save image under processed data

        io.imsave(outdir + f, img_as_ubyte(image))

        # flip image
        image_flipped = np.fliplr(image)
        io.imsave(outdir + fname + "_flipped.jpg", img_as_ubyte(image_flipped))

        # rotate image and save it
        rotate(image, outdir, fname, args.aug_per_image, args.max_rotation_angle)
        rotate(image_flipped, outdir, fname + "_flipped", args.aug_per_image, args.max_rotation_angle)

    print("Finished resizing and data augmentation...")
