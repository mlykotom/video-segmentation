import keras
import keras.backend as K
from keras import Input
from keras.applications import mobilenet
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization, Activation, concatenate, Conv2DTranspose, Reshape, Dropout, \
    MaxPooling2D, Lambda, SpatialDropout2D
from keras.models import Model

from base_model import BaseModel
from layers import BilinearUpSampling2D

from layers import tf_warp

"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.vis_utils import plot_model

from keras import backend as K


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


class MobileUNetV2(BaseModel):

    def _create_model(self):
        inputs = Input(shape=self.target_size + (3,), name='data_0')

        b00 = _conv_block(inputs, 32, (3, 3), strides=(2, 2))
        b01 = _inverted_residual_block(b00, 16, (3, 3), t=1, strides=1, n=1)
        b02 = _inverted_residual_block(b01, 24, (3, 3), t=6, strides=2, n=2)
        b03 = _inverted_residual_block(b02, 32, (3, 3), t=6, strides=2, n=3)
        b04 = _inverted_residual_block(b03, 64, (3, 3), t=6, strides=2, n=4)
        b05 = _inverted_residual_block(b04, 96, (3, 3), t=6, strides=1, n=3)
        b06 = _inverted_residual_block(b05, 160, (3, 3), t=6, strides=2, n=3)
        b07 = _inverted_residual_block(b06, 320, (3, 3), t=6, strides=1, n=1)

        # x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
        # x = GlobalAveragePooling2D()(x)
        # x = Reshape((1, 1, 1280))(x)
        # x = Dropout(0.3, name='Dropout')(x)
        # x = Conv2D((1, 1), padding='same')(x)
        #
        # x = Activation('softmax', name='softmax')(x)
        # output = Reshape((k,))(x)

        filters = 320

        up1 = concatenate([
            Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b07),
            b06,
        ], axis=3)

        up = up1

        # b14 = self._depthwise_conv_block(up1, filters, self.alpha_up, self.depth_multiplier, block_id=14)
        #
        # up = Conv2D(self.n_classes, (1, 1), kernel_initializer='he_normal', activation='linear')(b07)
        # up = BilinearUpSampling2D(size=(2, 2))(up)
        #
        # up = Reshape((-1, self.n_classes))(up)
        # up = Activation('softmax')(up)
        return Model(inputs, up)


if __name__ == '__main__':
    target_size = 256, 512
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = MobileUNetV2(target_size, 34)
    print(model.summary())
    keras.utils.plot_model(model.k, 'mobile_unet_v2.png', show_shapes=True, show_layer_names=True)
