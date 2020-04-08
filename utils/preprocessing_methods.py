import numpy as np
from skimage import io, img_as_ubyte, transform


def trim_image_rgb(jpg, dir, outdir):
    """Trims the black margins out of the image
    The original and returned images are rgb"""

    img = io.imread(dir+jpg)
    ts = (img != 0).sum(axis=1) != 0
    ts = ts.sum(axis=1) != 0
    img = img[ts]
    ts = (img != 0).sum(axis=0) != 0
    ts = ts.sum(axis=1) != 0
    img = img[:,ts,:]
    io.imsave(outdir + jpg, img)


def rotate(img, outdir, fname, n_aug, max_rotation_angle):
    # Prerequisites for rotating: Only those images should be rotated on which the retina is a 'whole' circle
    # They are defined by the distance between two black pixels (values of these pixels are (0, 0, 0)) in the first
    # and last row respectively the column
    # Max distance is needed and depends on the image size
    def check_prereq(array):
        l = len(array)
        max_distance = 0.20 * l
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
        rotation_angles = np.random.random_integers(low=-max_rotation_angle, high=max_rotation_angle, size=n_aug)
        for angle in rotation_angles:
            io.imsave(outdir + fname + "_rot_%i.jpg" % angle, img_as_ubyte(transform.rotate(img, angle)))