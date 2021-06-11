#!/usr/bin/env python

# Usage: python main.py learning_rate epochs batch_size model

import os
import sys

# import comet_ml at the top of your file
from comet_ml import Experiment

# How to use in leonhard:
# make sure you have all necessary packages installed!
# module load eth_proxy gcc/6.3.0 python_gpu/3.8.5 hdf5/1.10.1
# bsub -n 8 -W 12:00 -R "rusage[mem=4096, ngpus_excl_p=1]" python main.py
# bsub -n 8 -W 12:00 -R "rusage[mem=4096, ngpus_excl_p=1]"  -R "select[gpu_mtotal0>=16384]" python main.py

# print('Argument List:', str(sys.argv))
sys.path.append(os.getcwd())  # to run on clusters, make sure that local imports come after this

import tensorflow as tf
from EfficientNet400 import EfficientNet400
from EfficientNet256 import EfficientNet256
from EfficientNet128 import EfficientNet128
import Datagenerator
import loss_functions
from Datawriter import create_submission
from xception400 import xception400
from xception256 import xception256
from xception128 import xception128
from prexception import prexception
from preEfficientNet import preEfficientNet
import numpy as np
import random

# Create an experiment with your api key
experiment = Experiment(
    api_key="W5Oml4swMh30o4GyJFQ3068eC",
    project_name="RoadSegmentation",
    workspace="walon1998",
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
    "learning_rate": float(sys.argv[1]),
    "epochs": int(sys.argv[2]),
    "batch_size": int(sys.argv[3]),
    "model": sys.argv[4],
}
experiment.log_parameters(hyper_params)

# Sets seed for reproducibility
tf.random.set_seed(420)
np.random.seed(123)
random.seed(123)

# Create a MirroredStrategy for multi gpu training
# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.get_strategy()
print(strategy)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Check if GPU available and set memory growth instead of preallocating
# Preallocating can sometimes fail, note that memory growth does not work with MirroredStrategy
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(hyper_params["learning_rate"])
    model = None

    if hyper_params["model"] == "xception400":
        train_ds, test_ds, google_ds, normalization_dataset = Datagenerator.get_data(hyper_params["batch_size"], img_size_input=400, img_size_groundtruth=400, scale=True)
        model = xception400()
    elif hyper_params["model"] == "xception256":
        train_ds, test_ds, google_ds, normalization_dataset = Datagenerator.get_data(hyper_params["batch_size"], img_size_input=256, img_size_groundtruth=272, scale=True)
        model = xception256()
    elif hyper_params["model"] == "xception128":
        train_ds, test_ds, google_ds, normalization_dataset = Datagenerator.get_data(hyper_params["batch_size"], img_size_input=128, img_size_groundtruth=144, scale=True)
        model = xception128()
    elif hyper_params["model"] == "prexception":
        train_ds, test_ds, google_ds, normalization_dataset = Datagenerator.get_data(hyper_params["batch_size"], img_size_input=299, img_size_groundtruth=304, scale=True)
        model = prexception()
    elif hyper_params["model"] == "efficientnet400":
        train_ds, test_ds, google_ds, normalization_dataset = Datagenerator.get_data(hyper_params["batch_size"], img_size_input=400, img_size_groundtruth=400, scale=True)
        model = EfficientNet400(normalization_data=normalization_dataset)
    elif hyper_params["model"] == "efficientnet256":
        train_ds, test_ds, google_ds, normalization_dataset = Datagenerator.get_data(hyper_params["batch_size"], img_size_input=256, img_size_groundtruth=256, scale=True)
        model = EfficientNet256(normalization_data=normalization_dataset)
    elif hyper_params["model"] == "efficientnet128":
        train_ds, test_ds, google_ds, normalization_dataset = Datagenerator.get_data(hyper_params["batch_size"], img_size_input=128, img_size_groundtruth=128, scale=True)
        model = EfficientNet128(normalization_data=normalization_dataset)
    elif hyper_params["model"] == "preefficientnet":
        train_ds, test_ds, google_ds, normalization_dataset = Datagenerator.get_data(hyper_params["batch_size"], img_size_input=380, img_size_groundtruth=384, scale=False)
        model = preEfficientNet(normalization_data=normalization_dataset)
    else:
        print("Specified wrong model")
        exit(-1)

    model.compile(optimizer=optimizer,
                  loss=loss_functions.Semantic_loss_functions().dice_loss,
                  metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), ])

# print(model.summary())
# tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, to_file="model.png")
# exit(1)

# Train on google data
model_history = model.fit(google_ds, epochs=hyper_params["epochs"], verbose=1, validation_data=train_ds)

create_submission(model, test_ds, experiment)
