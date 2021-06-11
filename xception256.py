from keras.applications import Xception
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, MaxPooling2D, Dropout, concatenate, ZeroPadding2D, Conv2DTranspose
from keras.models import Model


# Unet which uses and Xception model 256
# Inspired by: https://www.kaggle.com/meaninglesslives/unet-xception-keras-for-pneumothorax-segmentation and https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py


def conv_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def res_block(x, filters):
    x_1 = LeakyReLU(alpha=.1)(x)
    x_1 = BatchNormalization()(x_1)
    x_2 = BatchNormalization()(x)
    x_1 = conv_block(x_1, filters, (3, 3))
    x_1 = conv_block(x_1, filters, (3, 3), activation=False)
    x = Add()([x_1, x_2])
    return x


def decoder_block(x, filters, kernel_size=(3, 3), activation=True):
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = res_block(x, filters)
    x = res_block(x, filters)
    if activation == True:
        x = LeakyReLU(alpha=.1)(x)
    return x


def xception256():
    backbone = Xception(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
    # tf.keras.utils.plot_model(backbone, show_shapes=True, expand_nested=False, to_file="xception.png")
    # print(backbone.summary())
    input = backbone.input

    conv4 = backbone.layers[121].output  # (None, 16, 16, 1024)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)  # (None, 8, 8, 1024)
    pool4 = Dropout(.5)(pool4)

    decoder4 = decoder_block(pool4, 512, (3, 3))  # (None, 8, 8, 512)
    deconv4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="valid")(decoder4)  # (None, 17, 17, 256)
    conv4 = ZeroPadding2D(((1, 0), (1, 0)))(conv4)  # (None, 17, 17, 1024)
    uconv4 = concatenate([deconv4, conv4])  # (None, 17, 17, 1280)
    uconv4 = Dropout(.5)(uconv4)  # (None, 17, 17, 1280)

    decoder3 = decoder_block(uconv4, 256, (3, 3))  # (None, 25, 25, 256)
    deconv3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(decoder3)  # (None, 50, 50, 128)
    conv3 = backbone.layers[31].output  # (None, 50, 50, 728)
    conv3 = ZeroPadding2D(((1, 1), (1, 1)))(conv3)  # (None, 17, 17, 1024)
    uconv3 = concatenate([deconv3, conv3])  # (None, 50, 50, 856)
    uconv3 = Dropout(.5)(uconv3)

    decoder2 = decoder_block(uconv3, 128, (3, 3))  # (None, 50, 50, 128)
    deconv2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(decoder2)  # (None, 100, 100, 64)
    conv2 = backbone.layers[21].output  # ((None, 99, 99, 256), 'block3_sepconv2_bn')
    conv2 = ZeroPadding2D(((3, 2), (3, 2)))(conv2)  # (None, 100, 100, 256)
    uconv2 = concatenate([deconv2, conv2])  # (None, 100, 100, 320)
    uconv2 = Dropout(.1)(uconv2)

    decoder1 = decoder_block(uconv2, 64, (3, 3))  # (None, 100, 100, 64)
    deconv1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(decoder1)  # (None, 200, 200, 32)
    conv1 = backbone.layers[11].output  # (None, 197, 197, 128)
    conv1 = ZeroPadding2D(((6, 5), (6, 5)))(conv1)  # (None, 200, 200, 128)
    uconv1 = concatenate([deconv1, conv1])  # (None, 200, 200, 160)
    uconv1 = Dropout(.1)(uconv1)

    decoder0 = decoder_block(uconv1, 32, (3, 3))  # (None, 200, 200, 32)
    uconv0 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(decoder0)  # (None, 400, 400, 16)
    uconv0 = Dropout(.5)(uconv0)

    final_decoder = decoder_block(uconv0, 16, (3, 3))  # (None, 400, 400, 16)
    final_decoder = Dropout(.25)(final_decoder)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(final_decoder)  # (None, 400, 400, 1)

    model = Model(input, output_layer)

    return model
