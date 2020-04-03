from tqdm import tqdm
import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
import argparse

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Resize Test-Images""")
    parser.add_argument('imdir', type=str, default=None, metavar='image_dir',
                        help="""The path to the directory which contains the image folder. """)
    parser.add_argument('outdir', type=str, default=None, metavar='image_dir',
                        help="""The path to the directory which should contain processed augmented images.""")
    parser.add_argument('-r', '--resize', type=int, default=[192, 188], nargs=2,
                        help="""Enter wished Size. Example use: -r 192 188""")
    parser.add_argument('-gr', '--grayscale', type=int, default=0,
                        help="""Grayscale the images. If wished enter an integer (except zero).""")
    args = parser.parse_args()

    dir = args.imdir
    outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    print("Start resizing...")
    for f in tqdm(os.listdir(dir)):
        fname = f.replace(".jpg", "")
        image = io.imread(outdir + f)

        # Resize image
        image = resize(image, output_shape=(args.resize[0], args.resize[1]))

        # save image under processed data
        io.imsave(outdir + f, img_as_ubyte(image))

    print("Finished resizing...")

