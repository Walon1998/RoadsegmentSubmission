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
predictions = ["8da84b87241d43d182a766c1d59ad18c"
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
