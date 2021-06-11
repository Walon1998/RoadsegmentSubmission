#!/usr/bin/env python3

import glob
import os
import re

import matplotlib.image as mpimg
import numpy as np


# Creates submission files from image masks
# From the official kaggle data

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > 0.25:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", os.path.basename(image_filename)).group(0))
    im = mpimg.imread(image_filename)
    im = np.round(im)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    print("Creating submission file")
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


if __name__ == '__main__':
    submission_filename = '2021.03.06.17.11.42_submission_round.csv'
    image_filenames = glob.glob("./Predictions/2021.03.06.17.11.42/*.png")
    print(image_filenames)
    masks_to_submission(submission_filename, *image_filenames)
