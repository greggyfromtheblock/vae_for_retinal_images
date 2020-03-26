from decode_diagnostics_keywords import decode_d_k
from tqdm import tqdm
from preprocessing_methods import trim_image_rgb, rotate, find_optimal_image_size_and_extend_db
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
    parser.add_argument('imdir', type=str, default=None,
                        metavar='image_dir',
                        help="""The path to the directory which contains
                       the image folder. """)
    parser.add_argument('outdir', type=str, default=None,
                        metavar='image_dir',
                        help="""The path to the directory which should contain
                        processed augmented images.""")
    parser.add_argument('xlsx_dir', type=str, default=None,
                        metavar='image_dir',
                        help="""The path to the directory which contains
                       the Annotations of the Odir-Dataset. """)
    args = parser.parse_args()

    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return(path)

    dir = add_slash(args.imdir)
    outdir = add_slash(args.outdir)
    xlsx_dir = add_slash(args.xlsx_dir)

    """
    python3 utils/preprocessing.py 
    /home/henrik/PycharmProjects/vae_for_retinal_images/data/odir/ODIR-5K_Training_Dataset/ 
    /home/henrik/PycharmProjects/vae_for_retinal_images/data/processed
    /home/henrik/PycharmProjects/vae_for_retinal_images/data/odir
    """

    os.makedirs(outdir, exist_ok=True)

    print("Start cropping...")
    for i, f in tqdm(enumerate(os.listdir(dir))):
        # Crop image
        trim_image_rgb(f, dir, outdir)
    print("Finished cropping...")

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
        rotate(image, outdir, fname)
        rotate(image_flipped, outdir, fname + "_flipped")

    print("Finished resizing and data augmentation...")

    print("Decode diagnostics keywords...")
    decode_d_k(xlsx_dir)
    print("Finished decoding diagnostics keywords...")

/data/analysis/ag-reils/ag-reils-shared-students/retina/data/raw/ODIR_Training_Images