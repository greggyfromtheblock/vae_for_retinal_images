from tqdm import tqdm
from preprocessing_methods import trim_image_rgb, rotate
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
    
    Subsequently, the augmentation step follows:
    Grayscale if stated, 
    resize to denoted size, 
    flip images,
    rotate those images whose retinas are complete circles.
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
    parser.add_argument('-r', '--resize', type=int, default=[192, 188], nargs=2,
                        help="""Enter wished Size. Example use: -r 192 188""")
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
        trim_image_rgb(f, dir, outdir)
    print("Finished cropping...")

    print("Start resizing and data augmentation...")
    for f in tqdm(os.listdir(outdir)):
        fname = f.replace(".jpg", "")
        image = io.imread(outdir + f)

        # Grayscale images if stated
        if args.grayscale:
            from skimage.color import rgb2gray
            image = rgb2gray(image)

        # Resize image
        image = resize(image, output_shape=(args.resize[0], args.resize[1]))

        # save image under processed data
        io.imsave(outdir + f, img_as_ubyte(image))

        if args.n_augmentation > 0:
            # flip image
            image_flipped = np.fliplr(image)
            io.imsave(outdir + fname + "_flipped.jpg", img_as_ubyte(image_flipped))

            # rotate image and save it
            rotate(image, outdir, fname, args.n_augmentation, args.max_rotation)
            rotate(image_flipped, outdir, fname + "_flipped", args.n_augmentation, args.max_rotation)

    print("Finished resizing and data augmentation...")

