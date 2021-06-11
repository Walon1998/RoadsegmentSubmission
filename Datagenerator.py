from functools import partial

import tensorflow as tf

# this file is responsible for loading data

# define how many threads to use to preprocess data
# NUM_THREADS = tf.data.AUTOTUNE # Only use if tf version >= 2.4
NUM_THREADS = 8  # or use tf.data.autotune (does not work on leanhard...)


# Decides input img
def decode_img(image, img_size, scale):
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_png(image)

    # resize if necessary
    if image.shape[0] != img_size or image.shape[1] != img_size:
        image = tf.image.resize(image, [img_size, img_size], antialias=True)

    # scale from [0,255] to [0,1]
    # Be aware than some neural networks do this on their own
    if scale:
        image = tf.cast(image, tf.float32) / 255.0

    return image


# loads and image
def load_image(img_size, scale, image_filepath):
    img = tf.io.read_file(image_filepath)
    img = decode_img(img, img_size, scale)
    return img


# Configure datasets from performance
def configure_for_performance(ds, batch_size):
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=NUM_THREADS)
    return ds


# Augments the dataset
# this function is applied each epoch, such that the network never sees the same picture twice
@tf.function
def augment_img(img, mask):
    p = 0.5
    # tf.print("img augmentation")

    if tf.random.uniform(()) > p:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > p:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    rotate_k = tf.random.uniform(shape=[], dtype=tf.dtypes.int32, maxval=4)  # 0 included, 4 excluded
    img = tf.image.rot90(img, k=rotate_k)
    mask = tf.image.rot90(mask, k=rotate_k)

    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.0, upper=0.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_saturation(img, lower=0.0, upper=0.5)

    return img, mask


# creates the used datasets
def get_data(batch_size, img_size_input, img_size_groundtruth, scale):
    # Read file name from dir
    train_ds = tf.data.Dataset.list_files("./Data/train_images/*", shuffle=False)
    test_ds = tf.data.Dataset.list_files("./Data/test_images/*", shuffle=False)
    train_groundtruth = tf.data.Dataset.list_files("./Data/groundtruth/*", shuffle=False)
    google_train_ds = tf.data.Dataset.list_files("./Data/Additional_Data/Images/*", shuffle=False)
    google_groundtruth_ds = tf.data.Dataset.list_files("./Data/Additional_Data/Masks/*", shuffle=False)

    # Load images
    train_ds = train_ds.map(partial(load_image, img_size_input, scale), num_parallel_calls=NUM_THREADS)
    test_ds_image = test_ds.map(partial(load_image, img_size_input, scale), num_parallel_calls=NUM_THREADS)

    train_groundtruth = train_groundtruth.map(partial(load_image, img_size_groundtruth, True), num_parallel_calls=NUM_THREADS)
    google_train_ds = google_train_ds.map(partial(load_image, img_size_input, scale), num_parallel_calls=NUM_THREADS)
    google_groundtruth_ds = google_groundtruth_ds.map(partial(load_image, img_size_groundtruth, True), num_parallel_calls=NUM_THREADS)

    normalization_dataset = train_ds.concatenate(google_train_ds)  # We need this dataset for normalization layers

    # combine train images and groundtruth
    train_ds = tf.data.Dataset.zip((train_ds, train_groundtruth))
    google_ds = tf.data.Dataset.zip((google_train_ds, google_groundtruth_ds))

    # combine test image and filename for later reconstruction
    test_ds = tf.data.Dataset.zip((test_ds_image, test_ds))

    # Cache before augmentation, otherwise the data is only augmented once
    train_ds = train_ds.cache()
    google_ds = google_ds.cache()
    test_ds = test_ds.cache()
    normalization_dataset = normalization_dataset.cache()

    # augment dataset
    google_ds = google_ds.map(augment_img, num_parallel_calls=NUM_THREADS)

    # configure datasets for performance
    train_ds = configure_for_performance(train_ds, batch_size)
    google_ds = configure_for_performance(google_ds, batch_size)
    test_ds = configure_for_performance(test_ds, batch_size)
    normalization_dataset = configure_for_performance(normalization_dataset, batch_size)

    return train_ds, test_ds, google_ds, normalization_dataset
