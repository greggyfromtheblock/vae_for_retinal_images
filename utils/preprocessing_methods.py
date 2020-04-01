import os
import random
import sys

import numpy as np
import pandas as pd
import PIL as pil
from skimage import img_as_ubyte, io, transform
from tqdm import tqdm


def trim_image_rgb(jpg, dir, outdir):
    """Trims the black margins out of the image
    The original and returned images are rgb"""

    img = io.imread(dir + jpg)
    ts = (img != 0).sum(axis=1) != 0
    ts = ts.sum(axis=1) != 0
    img = img[ts]
    ts = (img != 0).sum(axis=0) != 0
    ts = ts.sum(axis=1) != 0
    img = img[:, ts, :]
    io.imsave(outdir + jpg, img)


def find_optimal_image_size_and_extend_db(imdir="processed/train/"):
    """
    :param imdir: Directory to cropped images
    :return: Tuple: values for new Image Size
    """
    #imdir = db + imdir

    n = len(os.listdir(imdir)) #number of images
    x = np.zeros(n).astype("int")
    y = np.zeros(n).astype("int")
#    z = np.zeros(n).astype("int")
#    w = np.zeros(n).astype("int")

    min_x, min_y = float("inf"), float("inf")
    i = 0
    for f in tqdm(os.listdir(imdir)):
        img = pil.Image.open(imdir + '/' + f)
        x[i] = img.width
        y[i] = img.height
        i += 1
        if img.width * img.height < min_x * min_y:
            min_x, min_y = img.width, img.height
        img.close

    print("minimal/maximal size (width-height):", (min(x), min(y)), (max(x), max(y))),

    avg_size = np.sum(x) / np.alen(x), np.sum(y) / np.alen(x)
    print("Average size:     Width %i   Height %i" % (avg_size[0], avg_size[1]))

    print("Minimal/Maximal ratio (height/width):", min(y / x), max(y / x))

    ratio_height_width = np.sum(y) / np.sum(x)
    print("Total ratio (left_height/left_width):", ratio_height_width)

    print("\nMinimal image size: %ix%i\n" % (int(min_x), int(min_y)))

    for new_w in range(min_x, int(avg_size[0]), 2):
        for new_h in range(min_y, int(avg_size[1]), 2):
            if ratio_height_width - 0.01 < new_h / new_w < ratio_height_width + 0.01:
                print(
                    "Best minimal image size (close to the average ratio): %ix%i\n"
                    % (new_w, new_h)
                )
                return new_w, new_h


def rotate(img, outdir, fname, n_aug_per_image, max_rotation_angle):
    # Prerequisites for rotating: Only those images should be rotated on which the retina is a 'whole' circle
    # They are defined by the distance between two black pixels (values of these pixels are (0, 0, 0)) in the first
    # and last row respectively the column
    # Max distance is needed and depends on the image size
    def check_prereq(array):
        l = len(array)
        max_distance = 0.225 * l
        min, max = l, 0
        for i, el in enumerate(array):
            if i == 0 or i == l - 1:
                continue

            if i < l // 2:
                prev_el = array[i - 1]
                if (
                    el[0] != 0
                    or el[1] != 0
                    or el[2] != 0
                    and prev_el[0] == 0
                    and prev_el[1] == 0
                    and prev_el[2] == 0
                ):
                    min = i

            if i > l // 2:
                next_el = array[i + 1]
                if (
                    el[0] != 0
                    or el[1] != 0
                    or el[2] != 0
                    and next_el[0] == 0
                    and next_el[1] == 0
                    and next_el[2] == 0
                ):
                    max = i

        return abs(max - min) < max_distance

    if (
        check_prereq(img[0])
        and check_prereq(img[-1])
        and check_prereq(np.transpose(img)[0])
        and check_prereq(np.transpose(img)[-1])
    ):
        rotation_angles = [random.randrange(-max_rotation_angle, max_rotation_angle) for x in range(n_aug_per_image)]
        for angle in rotation_angles:
            io.imsave(outdir + fname + "_rot_%i.jpg" % angle, img_as_ubyte(transform.rotate(img, angle)))
