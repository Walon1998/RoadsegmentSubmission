import glob
import os
import re

import tensorflow as tf

from mask_to_submission import masks_to_submission


# this file is responsible for writing data / creating submissions

# Create a submission
def create_submission(model, test_ds, name):
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
            tf.io.write_file("./Predictions/{model_name}/{number}.png".format(model_name=name, number=number), png)
            # display([img, img_resized, img_int])

            image_counter += 1
            print(".", end='')
    # print(image_counter)

    # Create submission file
    submission_filename = 'Predictions/{}/{}_submission.csv'.format(name, name)
    image_filenames = glob.glob("./Predictions/{}/*.png".format(name))
    masks_to_submission(submission_filename, *image_filenames)

    print("Done", flush=True)
