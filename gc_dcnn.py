from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Concatenate, UpSampling2D
from tensorflow.keras.activations import sigmoid
from pyramid_pooling_module import PyramidPoolingModule
from tensorflow.keras import Model, Input

# model based on https://doi.org/10.1016/j.ins.2020.05.062

def residual_dilated_block(inp, filters):
    x = bn_relu(inp)
    x = Conv2D(filters=filters, kernel_size=3, strides=2, dilation_rate=1, padding='same')(x)

    x = bn_relu(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, dilation_rate=2, padding='same')(x)

    x = bn_relu(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, dilation_rate=2, padding='same')(x)
    # Add input back; residual
    # because of the stride size 2 of the first convolution layer we have to down-sample the input too
    y = Conv2D(filters=filters, kernel_size=1, strides=2, padding='same')(inp)
    x = Add()([x, y])
    return x


def encode(inp):
    size1, size2, size3, size4 = 64, 128, 256, 512
    x = Conv2D(filters=size1, kernel_size=3, padding='same')(inp)
    x = bn_relu(x)
    x = Conv2D(filters=size1, kernel_size=3, padding='same')(x)
    inp_hat = Conv2D(filters=size1, kernel_size=1, padding='same')(inp)
    x = Add()([x, inp_hat])
    skip1 = residual_dilated_block(x, size2)
    skip2 = residual_dilated_block(skip1, size3)
    skip3 = residual_dilated_block(skip2, size4)
    return skip1, skip2, skip3

def residual_block(inp, filters):
    x = bn_relu(inp)
    x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)

    x = bn_relu(x)
    x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)

    x = bn_relu(x)
    x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)

    # add the residual
    res = Conv2D(filters=filters, kernel_size=1, padding='same')(inp)
    x = Add()([x, res])
    return x


def bn_relu(x):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def decode(inp, skip1, skip2, skip3):
    size3, size2, size1 = 256, 128, 64
    x = Concatenate()([inp, skip3])
    x = UpSampling2D()(x)
    x = residual_block(x, size3)

    x = Concatenate()([x, skip2])
    x = UpSampling2D()(x)
    x = residual_block(x, size2)

    x = Concatenate()([x, skip1])
    x = UpSampling2D()(x)
    x = residual_block(x, size1)

    output = Conv2D(filters=1, kernel_size=1, activation=sigmoid)(x)
    return output


def gc_dcnn_model():
    input = Input(shape=(400, 400, 3))
    skip1, skip2, skip3 = encode(input)

    x = PyramidPoolingModule(padding='same', num_filters=512//4, pool_mode='max')(skip3)
    output = decode(x, skip1, skip2, skip3)
    return Model(inputs=input, outputs=output)
