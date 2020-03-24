from decode_diagnostics_keywords import decode_d_k
from tqdm import tqdm
from preprocessing_methods import trim_image_rgb, rotate, find_optimal_image_size_and_extend_db
import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
import numpy as np


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
    Flip images, Rotate those images whose retinas are ca complete circles 
    """

    dir = "/home/henrik/PycharmProjects/vae_for_retinal_images/data/"
    odir = 'odir/ODIR-5K_Training_Dataset/'
    outdir = 'processed/train/'

    os.makedirs(dir+outdir, exist_ok=True)
    """
     print("Start cropping...")
    for f in tqdm(os.listdir(dir+odir)):
        # Crop image
        trim_image_rgb(f, dir+odir, dir+outdir)
    print("Finished cropping...")
    """


    print("Start finding optimal image size and extend db...")
    opt_w, opt_h = find_optimal_image_size_and_extend_db(dir)
    print("Finished finding optimal image size and extend db...")

    print("Start resizing and data augmentation...")
    for f in tqdm(os.listdir(dir+outdir)):
        fname = f.replace(".jpg", "")
        image = io.imread(dir+outdir + f)

        # Resize image
        image = resize(image, output_shape=(opt_w, opt_h))

        # save image under processed data
        io.imsave(dir+outdir + f, img_as_ubyte(image))

        # flip image
        image_flipped = np.fliplr(image)
        io.imsave(dir+outdir + fname + "_flipped.jpg", img_as_ubyte(image_flipped))

        # rotate image and save it
        rotate(image, dir+outdir, fname)
        rotate(image_flipped, dir+outdir, fname + "_flipped")

    print("Finished resizing and data augmentation...")

    print("Decode diagnostics keywords...")
    decode_d_k(dir)
    print("Finished decoding diagnostics keywords...")

