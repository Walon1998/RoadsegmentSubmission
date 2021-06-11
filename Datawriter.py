import glob
import os
import re

import tensorflow as tf

from mask_to_submission import masks_to_submission


# this file is responsible for writing data / creating submissions
# In earlier version there was a callback, which could log validation images to tensorboard/comet, such that one could verify early on that the model learns something and works correctly
# There was also functionality to create a submission every epoch
# I found those methods especially useful in the early stage, but once you are sure that your model works correctly there is not much use for them, so I removed them.
# If you still need them you can create them by checking out an earlier version of this file.

# Create a submission
def create_submission(model, test_ds, experiment):
    # Prediction of test dataset
    result = model.predict(test_ds)

    # Save masks of predictions of test dataset
    image_counter = 0
    for batch_img, batch_file_name in test_ds:
        for file_name in batch_file_name:
            basename = os.path.basename(file_name.numpy())
            number = int(re.sub("\D", "", str(basename)))
            img = result[image_counter]
            img_resized = tf.image.resize(img, [608, 608], antialias=True)
            img_int = tf.cast(img_resized * 255.0, tf.uint8)
            png = tf.io.encode_png(img_int)
            tf.io.write_file("./Predictions/{model_name}/{number}.png".format(model_name=experiment.get_key(), number=number), png)
            # display([img, img_resized, img_int])

            image_counter += 1
            print(".", end='')
    # print(image_counter)

    # Create submission file
    submission_filename = 'Predictions/{}/{}_submission.csv'.format(experiment.get_key(), experiment.get_key())
    image_filenames = glob.glob("./Predictions/{}/*.png".format(experiment.get_key()))
    masks_to_submission(submission_filename, *image_filenames)

    # Log stuff to comet
    experiment.log_asset_folder(os.path.dirname(submission_filename))
    experiment.log_code("./Datagenerator.py")
    experiment.log_code("./Datawriter.py")
    experiment.log_code("./loss_functions.py")
    experiment.log_code("./google_maps_gen.py")
    experiment.log_code("./mask_to_submission.py")
    experiment.log_code("./xception400.py")
    experiment.log_code("./xception256.py")
    experiment.log_code("./xception128.py")
    experiment.log_code("./EfficientNet400.py")
    experiment.log_code("./EfficientNet256.py")
    experiment.log_code("./EfficientNet128.py")
    experiment.log_code("./preEfficientNet.py")
    experiment.log_code("./prexception.py")

    print("Done", flush=True)
