from skimage import io, img_as_ubyte
from skimage.transform import resize
import os
import matplotlib.pyplot as plt
import numpy as np


# Cropping the black area of the images and save images in a new directory
# Each Pixel in this area is defined as (0,0,0)
def cropping(jpg):
    path = "/home/henrik/PycharmProjects/Project A - VAE Retina/ODIR-5K_Training_Images/"
    odir = "ODIR-5K_Training_Dataset/"
    cropped = "cropped_Images/"

    # Change directory
    os.chdir(path + odir)
    img = io.imread(jpg, plugin='matplotlib')

    # Remember minimals and maximal coordinates where there is no black area
    minx, maxx, miny, maxy = img.shape[1], 0, img.shape[0], 0  # Width: x-axis, Height: y-axis

    # for each element in image matrix
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] != 0 and img[i, j, 1] != 0 and img[i, j, 2] != 0:
                if i < miny:
                    miny = i
                elif i > maxy:
                    maxy = i
                if j < minx:
                    minx = j
                elif j > maxx:
                    maxx = j

    # save image under cropped_Images
    os.chdir(path + cropped)
    cropped_img = img[miny:maxy, minx:maxx, :]

    # Resize image to reduce the complexity within cropping.py
    cropped_img = img_as_ubyte(resize(cropped_img, (256, 256, 3)))

    # e.g.: 0000_left.jpg --> slicing .jpg --> receiving  0000_left_cropping.jpg
    io.imsave(jpg[0:-4]+"_cropped.jpg", cropped_img)
    # plt.imshow(cropped_img)
    # plt.savefig(jpg[0:-4] + "_cropped_flipped.jpg")

    # Flip image
    flipped_img = np.fliplr(cropped_img)

    # e.g.: 0000_left.jpg --> slicing .jpg --> receiving  0000_left_cropping.jpg
    io.imsave(jpg[0:-4]+"_cropped_flipped.jpg", flipped_img)
    # plt.imshow(flipped_img)
    # plt.savefig(jpg[0:-4] + "_cropped_flipped.jpg")

    """
    # Displaying right image and flipped image to see if it works
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(cropped_img)
    plt.subplot(1, 3, 3)
    plt.imshow(flipped_img)
    plt.show()
    """

"""
# To see if it works
path = "/home/henrik/PycharmProjects/Project A - VAE Retina/ODIR-5K_Training_Images/"
odir = "ODIR-5K_Training_Dataset"

# Change directory
os.chdir(path + odir)

# List of strings which contains the file name of each image
jpg_list = os.listdir().copy()

cropping(jpg_list[0])
"""




