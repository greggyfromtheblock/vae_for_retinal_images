# vae_for_retinal_images
Variational Autoencoder

## Some Guidelines and Principles to use

Keep the code as modular as possible.

Don't use command line arguments. Put all the arguments in the config.json.

Don't use fixed path 'variables' like `xsl_file = $HOME/vae/foo.xsl`, they may
break the code or make it behave unexpectedly on different environment.

Since we are building a VAE we don't assume existance of labels and annotations
in the input. So preprocessing should only be done on the images themselves.
Preprocessing of the annotation files should be done separately.

Before you push code to your branch, run black on it. This can catch syntax
error and reformats your code to look more professional and legible:

```
# install black on your local python environment, for example if you use conda:
(your_conda_env)$> conda install black

# then:
black your_file.py

# or if you use vim, type in normal mode:
:!black % <ENTER>
```


## Application Working Principles
The scripts that are going to be executed from the command line are:

```
APP/train_model.py
APP/utils/preprocessing.py
APP/utils/preprocess_annotations.py

# step 1: annotations: (currently it uses commandline arguments but we may later
                        # move them to config.json....
                        # This script is independent of the two others
                        # Creates out the odir excel file a csv file which
                        # annotations for left/right funduses on separate lines

$> python utils/preprocess_annotations.py path/to/annotation.xsl path/to/output.csv

# step 2: preprocessing the images. currently it uses the commandline arguments.
                        # has optional augmentation by rotation/reflection
                        # doesn't change the raw data.
                        # currently has too many command line args...

$> python utils/preprocessing.py  **args


# step 3: run the training script. 
                    # should have no command line argumens except perhaps the
                    # path to the config.json file

$> python train_model.py
```


## Merging Branches (Work in Progress)
### train_model.py
What is currently on master branch,
can be safely replaced with train_model.py on hernrik's brach. The only
difference is that henrik branch calls for to setup funtion at the start to
parse the config.json file:


```
    FLAGS, logger = setup(running_script="./utils/training.py", config="config.json")
    print("FLAGS= ", FLAGS)
```

And then replaces all command line arguments with 
config file arguments which is what we want.


### utils/preprocess_annotations.py
It's a new file so there are no conflicts.
then remove from master 'utils/decode_diagnostics_keywords.py' by:
`git rm utils/decode_diagnostics_keywords.py`

### utils/preprocessing_methods.py
I see that the changes from yiftach branch have been merge so now master branch
is the latest. The main point to make sure is that the call for the function 
`find_optimal_image_size_and_extend_db` on the preprocessing.py is changed
because this function has been rewritten to take only one input argument
(imdir).

another thing is to see what is the issue with the rotate function.

### utils/preprocessing.py

This one looks like a mess but I think it should be based on yiftach branch and
just add the parts with 'max_rotation_angle'. So I suggest the following code
for master branch:

Unless there is a bug in the new rotate function I think the script below should
work because I tried my branch (yiftach) and it did run errorless.

```
# utils/preprocessing.py

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
                    ###### additional augmentation arguments
    parser.add_argument('aug_per_image', type=int, default=0,
                        metavar='n_augmentation',
                        help="""Number of Augmented images per image""")
    parser.add_argument('max_rotation_angle', type=int, default=0,
                        metavar='maximum_rotation_angle',
                        help="""Max rotation degree +- for the images, for example if you pass 10 to
                        this argument then the function will pick {aug_per_image} random values from
                        the range -10 to 10""")


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
            imdir = ddir + outdir)
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
        # rotate(image, ddir + outdir, fname) # old version
        #rotate(image_flipped, ddir + outdir, fname + "_flipped") #old version
        # rotate image and save it
        rotate(image, outdir, fname, args.aug_per_image, args.max_rotation_angle)
        rotate(image_flipped, outdir, fname + "_flipp

    print("Finished resizing and data augmentation...")

```

