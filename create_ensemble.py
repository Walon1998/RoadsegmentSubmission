#!/usr/bin/env python

import glob
import os

import numpy
import numpy as np
from PIL import Image

from mask_to_submission import masks_to_submission

# Use this file to create an ensemble of multiple predictions.
# You can specify the directories in the array "predictions"
# All images will be averaged across the experiment and a new submission will be created.

# the predictions key of which you want to create an ensemble


predictions = ["8da84b87241d43d182a766c1d59ad18c",
               "5564a7c45d9e4288b85bcb5bfa618373",
               "7f1e3a4337ff41aca66aeb6c8d0f5b98",
               'cc85e7e9ace9471eb8d0b1f97f04ae5f',
               '3dee64c3992c419ea94eb5afecc057e3',
               'c4c51ab7513248809a55574a70483db4',
               '8b4a5b2af7a8441fa8aac3647d0b19db',
               'a83b65ab64204b5f97432238dfa4f665',
               '7199da80a2a849ee9f6269812b1da412',
               'b98a5e50ec5a475e886af20151d363dd',
               'af619bc1bbcc4d69b24145c077514c74',
               '3d288100cfb741b89a980b1d55a46da1',
               '4959c8b9146f4a16923815d4a0dbe122',
               '2139f56d5c3340f3835e6d43b5737ca7',
               'c30e457b31e04a61a2a00bc9d42c248c',
               'c63847b4be5a467ba0cfecc08c73287a',
               'a5c30ae6d161428f9c13c72929387bb2',
               '48ecf36ec82643f2ab77c0293b9e3295',
               '19b909be440a4abdbe9aab88fae2018f',
               '0482898780804e379865258a89f1df34',
               ]

print("\nCreating Ensemble", end="")
images = glob.glob("Predictions/" + predictions[0] + "/*.png")  # all the images filelist

# create name of ensemble, combine the experiment_keys and foreground threshold
model_name = "Ensemble"
for name in predictions:
    model_name += "_" + name[0:4]

os.mkdir("Predictions/" + model_name)

# average the images
for image in images:
    image = os.path.basename(image)
    print(".", end='')  # to track progress
    arr = np.zeros((608, 608), np.float)

    for pred in predictions:
        imarr = np.array(Image.open("Predictions/" + pred + "/" + image), dtype=numpy.float)
        arr = arr + imarr

    arr = arr / len(predictions)
    arr = numpy.array(numpy.round(arr), dtype=numpy.uint8)
    out = Image.fromarray(arr)
    out.save("Predictions/" + model_name + "/" + image)

print("")
# Create submission file
submission_filename = 'Predictions/{}/{}_submission.csv'.format(model_name, model_name)
image_filenames = glob.glob("./Predictions/{}/*.png".format(model_name))
masks_to_submission(submission_filename, *image_filenames)
