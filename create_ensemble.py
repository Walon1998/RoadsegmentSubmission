#!/usr/bin/env python

import glob
import numpy
import os
import urllib.request

import numpy as np
from PIL import Image
from comet_ml.api import API

from mask_to_submission import masks_to_submission

# Use this file to create an ensemble of multiple predictions.
# You can specify the experiments using the experiment key.
# The code will then download all the images associated with the experiments. (Be aware that this does not work correctly if you save more images in comet.ml than the final predictions.)
# All images will be averaged across the experiment and a new submission will be created.

# the predictions key of which you want to create an ensemble
predictions = ["8da84b87241d43d182a766c1d59ad18c"
               ]

# my comet.ml api, you should use your own!
comet_api = API(api_key="W5Oml4swMh30o4GyJFQ3068eC")

print("Downloading images")  # Download all images from all the experiments
for experiment_key in predictions:
    print(experiment_key)
    experiment = comet_api.get_experiment(workspace="walon1998", project_name="RoadSegmentation", experiment=experiment_key)  # change thsi according to your project
    asset_list = experiment.get_asset_list(asset_type="image")

    dirname = "./Predictions/" + experiment_key  # save images in predictions dir
    if os.path.exists(dirname):
        continue
    os.mkdir(dirname)

    # download the images
    for asset in asset_list:
        print(".", end='')
        urllib.request.urlretrieve(asset["link"], dirname + "/" + asset["fileName"])
    print("")

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
