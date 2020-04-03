from tqdm import tqdm
from preprocessing_methods import trim_image_rgb, rotate, find_optimal_image_size
import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
import numpy as np
import argparse

# Ignore warnings
import warnings
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
    parser.add_argument('imdir', type=str, default=None, metavar='image_dir',
                        help="""The path to the directory which contains the image folder. """)
    parser.add_argument('outdir', type=str, default=None, metavar='out_image_dir',
                        help="""The path to the directory which should contain processed augmented images.""")
    parser.add_argument('-na', '--n_augmentation', type=int, default=0,
                        help="""Number of Augmented images per image""")
    parser.add_argument('-mra', '--max_rotation', type=int, default=0,
                        help="""Max rotation degree +- for the images, for example if you pass 10 to this argument then 
                        the function will pick {aug_per_image} random values from the range -10 to 10""")
    parser.add_argument('-gr', '--grayscale', type=int, default=0,
                        help="""Grayscale the images. If wished enter an integer (except zero).""")
    args = parser.parse_args()

    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return(path)

    dir = add_slash(args.imdir)
    outdir = add_slash(args.outdir)

    os.makedirs(outdir, exist_ok=True)

    print("Start cropping...")
    for i, f in tqdm(enumerate(os.listdir(dir))):
        # Crop image
        if i < 50:
            trim_image_rgb(f, dir, outdir)
    print("Finished cropping...")

    print("Start finding optimal image size and extend db...")
    opt_w, opt_h = find_optimal_image_size(imdir=outdir)
    print("Finished finding optimal image size and extend db...")

    print("Start resizing and data augmentation...")
    for f in tqdm(os.listdir(outdir)):
        fname = f.replace(".jpg", "")
        image = io.imread(outdir + f)

        # Check grayscale images
        if args.grayscale:
            from skimage.color import rgb2gray
            image = rgb2gray(image)

        # Resize image
        image = resize(image, output_shape=(opt_w, opt_h))

        # save image under processed data
        io.imsave(outdir + f, img_as_ubyte(image))

        # flip image
        image_flipped = np.fliplr(image)
        io.imsave(outdir + fname + "_flipped.jpg", img_as_ubyte(image_flipped))

        # rotate image and save it
        rotate(image, outdir, fname, args.n_augmentation, args.max_rotation)
        rotate(image_flipped, outdir, fname + "_flipped", args.n_augmentation, args.max_rotation)

    print("Finished resizing and data augmentation...")

