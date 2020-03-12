from skimage import io, img_as_ubyte
from skimage.transform import resize, rotate
import os
import matplotlib.pyplot as plt
import numpy as np


# Cropping the black (uninteresting)  area of the images as far as possible and save images in a new directory
# Each Pixel in this area is defined as (0,0,0)
def rotate_img(jpg):
    path = "/home/henrik/PycharmProjects/Project A - VAE Retina/ODIR-5K_Training_Images/"
    odir = "ODIR-5K_Training_Dataset/"
    cropped = "cropped_Images/"

    # Change directory
    os.chdir(path + cropped)
    img = io.imread(jpg, plugin='matplotlib')
	################################################################
	# The incoming images were previously resized to shape 256x256x3
	################################################################

    # Prerequisites for rotating: Only those images should be rotated on which the retina is a 'whole' circle
    # They are defined by the distance between two black pixels (values of these pixels are (0, 0, 0)) in the first
    # and last row respectively the column
    # Max distance is needed and depends on the image size
    def check_prereq(array, max_distance=70):
        array.flatten()
        l = len(array)
        min, max = l, 0
        for i, el in enumerate(array):
            if i == 0 or i == l-1:
                continue

            if i < l //2:
                prev_el = array[i - 1]
                # print(i, "prevEL", prev_el)
                if el[0] != 0 or el[1] != 0 or el[2] != 0 and prev_el[0] == 0 and prev_el[1] == 0 and prev_el[2] == 0:
                    min = i

            if i > l//2:
                next_el = array[i + 1]
                # print(i, "NextEl", next_el)
                if el[0] != 0 or el[1] != 0 or el[2] != 0 and next_el[0] == 0 and next_el[1] == 0 and next_el[2] == 0:
                    max = i

        # print(max,min)
        return True if abs(max-min) < max_distance else False

    if check_prereq(img[0]) and check_prereq(img[-1]) and check_prereq(np.transpose(img)[0]) and \
        check_prereq(np.transpose(img)[-1]):

        angles = [-35, -25, -20, -10, 15, 20, 30]  # should be still discussed
        for angle in angles:
            io.imsave(jpg[0:-4] + "_rot_%i.jpg" % angle, img_as_ubyte(rotate(img, angle)))

            """
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(rotated)
            plt.show()
            """

# To see if it works
path = "/home/henrik/PycharmProjects/Project A - VAE Retina/ODIR-5K_Training_Images/"
odir = "cropped_Images"

# Change directory
os.chdir(path + odir)

# List of strings which contains the file name of each image
jpg_list = os.listdir().copy()
jpg_list.sort()

for i, jpg in enumerate(jpg_list):
    rotate_img(jpg)
    print(jpg)




